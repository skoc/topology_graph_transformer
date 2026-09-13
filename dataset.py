"""Tissue-graph dataset: slide CSV + precomputed DGL graphs and topology caches."""
import glob
import os

import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

TYPING_LABELS = {
    "lung": {"LUAD": 0, "LUSC": 1},
    "kidney": {"KICH": 0, "KIRC": 1, "KIRP": 2},
}
STAGE_LABELS = {"I": 0, "II": 1, "III": 2, "IV": 3}
COHORTS = {"lung": "TCGA-LUNG", "kidney": "TCGA-KIDNEY"}


def load_slides(csv_path, cohort="lung", task="typing"):
    """Slide table for one cohort with a `label` column for the given task."""
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if cohort not in COHORTS:
        raise ValueError(f"cohort must be one of {sorted(COHORTS)}, got {cohort!r}")
    df = df[df["cohort"] == COHORTS[cohort]].copy()

    if task == "typing":
        mapping = TYPING_LABELS[cohort]
        df = df[df["subtype"].isin(mapping)]
        df["label"] = df["subtype"].map(mapping)
    elif task == "staging":
        df = df[df["stage"].isin(STAGE_LABELS)]
        df["label"] = df["stage"].map(STAGE_LABELS)
    else:
        raise ValueError(f"task must be 'typing' or 'staging', got {task!r}")

    if df.empty:
        raise ValueError(f"no slides left for cohort={cohort!r} task={task!r}")
    return df.reset_index(drop=True)


def patient_folds(df, n_folds=5, seed=42):
    """Stratified folds with all slides of a patient kept in one fold."""
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(df, df["label"], groups=df["patient_id"]))


def resolve_graph(graph_dir, slide_id):
    """Graph filenames keep the TCGA UUID, so match on the slide id prefix."""
    hits = sorted(glob.glob(os.path.join(graph_dir, f"{slide_id}*.bin")))
    return hits[0] if hits else None


class TissueGraphDataset(Dataset):
    """Node features are the 512-d ResNet descriptor plus normalised centroids.

    Slides whose graph is missing on disk are dropped at construction time and
    reported, so a partially built graph directory still trains.
    """

    def __init__(self, df, graph_dir, ph_path=None, toposcan_path=None, merge_path=None):
        self.rows = []
        self.keep_idx = []
        missing = 0
        for i, row in enumerate(df.itertuples()):
            path = resolve_graph(graph_dir, row.slide_id)
            if path is None:
                missing += 1
                continue
            self.rows.append((row.slide_id, path, int(row.label)))
            self.keep_idx.append(i)
        self.missing = missing
        if missing:
            print(f"skipped {missing} slides without a graph in {graph_dir}")
        if not self.rows:
            raise RuntimeError(f"no graphs found in {graph_dir}")

        self.ph = np.load(ph_path) if ph_path else None
        self.toposcan = np.load(toposcan_path) if toposcan_path else None
        self.merge = np.load(merge_path) if merge_path else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        slide_id, path, label = self.rows[idx]
        graph = load_graphs(path)[0][0]

        feat = graph.ndata["feat"].float()
        centroid = graph.ndata["centroid"].float()
        centroid = centroid / centroid.max().clamp(min=1.0)
        x = torch.cat([feat, centroid], dim=1)

        item = {"x": x, "label": label, "adj": self._adjacency(graph)}
        if self.toposcan is not None:
            item["toposcan"] = torch.from_numpy(self.toposcan[slide_id]).float()
        if self.merge is not None:
            item["merge"] = torch.from_numpy(self.merge[slide_id]).float()
        if self.ph is not None:
            item["ph"] = torch.from_numpy(self.ph[slide_id]).float()
        return item

    @staticmethod
    def _adjacency(graph):
        n = graph.num_nodes()
        src, dst = graph.edges()
        adj = torch.zeros(n, n)
        adj[src.long(), dst.long()] = 1.0
        adj[dst.long(), src.long()] = 1.0
        adj.fill_diagonal_(0.0)
        return adj


def collate(batch):
    """Pad a batch of variable-size graphs to the largest node count."""
    b = len(batch)
    n = max(item["x"].size(0) for item in batch)

    out = {
        "x": torch.zeros(b, n, batch[0]["x"].size(1)),
        "node_mask": torch.zeros(b, n, dtype=torch.bool),
        "adj": torch.zeros(b, n, n),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
    }
    if "toposcan" in batch[0]:
        t = batch[0]["toposcan"].shape[1]
        out["toposcan"] = torch.zeros(b, n, t, 4)
    if "merge" in batch[0]:
        out["merge"] = torch.zeros(b, n, n)
    if "ph" in batch[0]:
        out["ph"] = torch.stack([item["ph"] for item in batch])

    for i, item in enumerate(batch):
        ni = item["x"].size(0)
        out["x"][i, :ni] = item["x"]
        out["node_mask"][i, :ni] = True
        out["adj"][i, :ni, :ni] = item["adj"]
        if "toposcan" in out:
            out["toposcan"][i, :ni] = item["toposcan"]
        if "merge" in out:
            out["merge"][i, :ni, :ni] = item["merge"]
    return out
