"""Train TC-GT on one cross-validation fold.

    python train.py --csv data/slides.csv --graph_dir /path/to/bins \
        --task typing --cohort lung --fold 0 --use_ph --use_lta --use_bias
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from dataset import TissueGraphDataset, collate, load_slides, patient_folds
from model import TCGT


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train TC-GT")
    p.add_argument("--csv", default="data/slides.csv")
    p.add_argument("--graph_dir", required=True, help="directory of DGL .bin tissue graphs")
    p.add_argument("--task", default="typing", choices=["typing", "staging"])
    p.add_argument("--cohort", default="lung", choices=["lung", "kidney"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n_folds", type=int, default=5)

    p.add_argument("--use_ph", action="store_true", help="graph-level descriptor g_PH")
    p.add_argument("--use_bias", action="store_true", help="topology-aware attention")
    p.add_argument("--use_lta", action="store_true", help="layerwise topology aggregation")

    p.add_argument("--ph_dir", default="data/ph")
    p.add_argument("--local_topo_dir", default="data/local_topo")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=546)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=20, help="early stopping on val macro-F1")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="runs")
    return p.parse_args(argv)


def build_loaders(args):
    df = load_slides(args.csv, cohort=args.cohort, task=args.task)
    folds = patient_folds(df, n_folds=args.n_folds, seed=args.seed)
    train_idx, test_idx = folds[args.fold]

    dataset = TissueGraphDataset(
        df,
        args.graph_dir,
        ph_path=os.path.join(args.ph_dir, f"{args.cohort}_ph.npz") if args.use_ph else None,
        toposcan_path=(
            os.path.join(args.local_topo_dir, f"{args.cohort}_toposcan.npz")
            if args.use_lta
            else None
        ),
        merge_path=(
            os.path.join(args.local_topo_dir, f"{args.cohort}_merge.npz")
            if args.use_bias
            else None
        ),
    )

    remap = {orig: new for new, orig in enumerate(dataset.keep_idx)}

    def remap_split(indices):
        return [remap[int(i)] for i in indices if int(i) in remap]

    train_idx = remap_split(train_idx)
    test_idx = remap_split(test_idx)

    # 15% of the training split is held out for early stopping
    rng = np.random.default_rng(args.seed)
    train_idx = rng.permutation(train_idx)
    n_val = max(1, int(0.15 * len(train_idx)))
    val_idx, train_idx = train_idx[:n_val], train_idx[n_val:]

    def loader(indices, shuffle):
        return DataLoader(
            Subset(dataset, [int(i) for i in indices]),
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
        )

    n_classes = int(df["label"].max()) + 1
    return loader(train_idx, True), loader(val_idx, False), loader(test_idx, False), n_classes


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses, probs, targets = [], [], []

    with torch.set_grad_enabled(train):
        for batch in loader:
            inputs = {
                k: batch[k].to(device)
                for k in ("x", "node_mask", "adj", "merge", "toposcan", "ph")
                if k in batch
            }
            labels = batch["label"].to(device)

            logits = model(**inputs)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            probs.append(logits.softmax(-1).detach().cpu())
            targets.append(labels.cpu())

    return float(np.mean(losses)), torch.cat(probs).numpy(), torch.cat(targets).numpy()


def metrics(probs, targets):
    preds = probs.argmax(1)
    out = {
        "balacc": balanced_accuracy_score(targets, preds),
        "f1m": f1_score(targets, preds, average="macro"),
    }
    try:
        if probs.shape[1] == 2:
            out["auc"] = roc_auc_score(targets, probs[:, 1])
        else:
            out["auc"] = roc_auc_score(targets, probs, multi_class="ovr", average="macro")
    except ValueError:  # a fold may not contain every class
        out["auc"] = float("nan")
    return out


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader, test_loader, n_classes = build_loaders(args)
    device = torch.device(args.device)

    model = TCGT(
        num_classes=n_classes,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_ph=args.use_ph,
        use_bias=args.use_bias,
        use_lta=args.use_lta,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    )

    out_dir = os.path.join(args.out_dir, f"{args.cohort}_{args.task}_fold{args.fold}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "best.pt")

    best_f1, since_best = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, probs, targets = run_epoch(model, val_loader, device, criterion)
        val = metrics(probs, targets)
        scheduler.step(val["f1m"])

        print(
            f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
            f"balacc {val['balacc']:.4f}  f1m {val['f1m']:.4f}"
        )

        if val["f1m"] > best_f1:
            best_f1, since_best = val["f1m"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            since_best += 1
            if since_best >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, probs, targets = run_epoch(model, test_loader, device, criterion)
    test = metrics(probs, targets)
    print(f"test  balacc {test['balacc']:.4f}  f1m {test['f1m']:.4f}  auc {test['auc']:.4f}")

    np.savez(os.path.join(out_dir, "test_predictions.npz"), probs=probs, targets=targets)
    return test


if __name__ == "__main__":
    main()
