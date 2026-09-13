"""CSV parsing, patient grouping, and typing label maps."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import load_slides, patient_folds, TYPING_LABELS, STAGE_LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "slides.csv")


def test_typing_labels():
    lung = load_slides(CSV, cohort="lung", task="typing")
    kidney = load_slides(CSV, cohort="kidney", task="typing")
    assert set(lung["subtype"]) == set(TYPING_LABELS["lung"])
    assert set(kidney["subtype"]) == set(TYPING_LABELS["kidney"])
    assert set(lung["label"]) == {0, 1}
    assert set(kidney["label"]) == {0, 1, 2}
    assert lung["slide_id"].str.contains(r"\.").sum() == 0
    assert kidney["slide_id"].str.contains(r"\.").sum() == 0


def test_patient_folds():
    df = load_slides(CSV, cohort="lung", task="typing")
    folds = patient_folds(df, n_folds=5, seed=42)
    assert len(folds) == 5
    patients = df["patient_id"].values
    for train_idx, test_idx in folds:
        train_p = set(patients[train_idx])
        test_p = set(patients[test_idx])
        assert train_p.isdisjoint(test_p)
        assert len(train_idx) + len(test_idx) == len(df)


def test_staging_lung_only():
    lung = load_slides(CSV, cohort="lung", task="staging")
    assert set(lung["stage"]).issubset(STAGE_LABELS)
    try:
        load_slides(CSV, cohort="kidney", task="staging")
        raise AssertionError("kidney staging should be empty")
    except ValueError:
        pass


if __name__ == "__main__":
    test_typing_labels()
    test_patient_folds()
    test_staging_lung_only()
    print("ok")
