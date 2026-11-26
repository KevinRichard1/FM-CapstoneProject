"""Create the subject-level ADFTD-RS split manifest.

Subjects are assigned before any window is selected. The split is stratified
within AD, FTD, and HC and asserted to be disjoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_subject_labels(eeg_dir: Path, sample_rate_hz: int = 200) -> pd.DataFrame:
    meta = json.loads((eeg_dir / "meta.json").read_text(encoding="utf-8"))
    n = meta["N"]
    y = np.memmap(eeg_dir / "y.dat", dtype="float32", mode="r", shape=(n, 3))
    rates = np.rint(y[:, 2]).astype(int)
    labels = np.rint(y[:, 0]).astype(int)
    subjects = np.rint(y[:, 1]).astype(int)
    keep = rates == sample_rate_hz
    frame = pd.DataFrame(
        {"subject_id": subjects[keep], "disease_id": labels[keep]}
    )
    return frame.groupby("subject_id", as_index=False)["disease_id"].first()


def split_subjects(subject_frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Split subjects 60/20/20 within each diagnostic class."""
    parts: list[pd.DataFrame] = []
    for disease_id, group in subject_frame.groupby("disease_id", sort=True):
        subjects = group["subject_id"].to_numpy()
        train, remaining = train_test_split(
            subjects, test_size=0.4, random_state=seed, shuffle=True
        )
        val, test = train_test_split(
            remaining, test_size=0.5, random_state=seed, shuffle=True
        )
        for split, ids in (("train", train), ("val", val), ("test", test)):
            parts.append(
                pd.DataFrame(
                    {
                        "subject_id": sorted(int(value) for value in ids),
                        "disease_id": int(disease_id),
                        "split": split,
                    }
                )
            )
    manifest = pd.concat(parts, ignore_index=True).sort_values(
        ["split", "disease_id", "subject_id"]
    )
    return manifest.reset_index(drop=True)


def validate_manifest(manifest: pd.DataFrame, expected_subjects: set[int]) -> None:
    split_sets = {
        split: set(manifest.loc[manifest["split"] == split, "subject_id"])
        for split in ("train", "val", "test")
    }
    assert set(manifest["subject_id"]) == expected_subjects, "Subject coverage mismatch"
    assert len(manifest) == len(expected_subjects), "Duplicate subject assignments"
    assert split_sets["train"].isdisjoint(split_sets["val"]), "Train/val subject overlap"
    assert split_sets["train"].isdisjoint(split_sets["test"]), "Train/test subject overlap"
    assert split_sets["val"].isdisjoint(split_sets["test"]), "Val/test subject overlap"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eeg-dir", default="data/ADFTD-RS")
    parser.add_argument("--output", default="data/manifests/eeg_subject_split.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subject_frame = load_subject_labels(Path(args.eeg_dir))
    manifest = split_subjects(subject_frame, args.seed)
    validate_manifest(manifest, set(subject_frame["subject_id"].astype(int)))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(manifest.groupby(["split", "disease_id"]).size().to_string())
    print(f"Saved subject split to {output} (seed={args.seed})")


if __name__ == "__main__":
    main()
