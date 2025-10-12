# make_splits_C.py — build CNMR dataset once and save fixed train/val/test splits (repo-root relative)

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset

# ---------- Repo-root bootstrap (imports & paths work from anywhere) ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.GEOM_graph_builder import build_dataset, MatchedSplitConfig


# ---------- Simple wrapper ----------
class SimpleDataset(TorchDataset):
    def __init__(self, data_list): self.data_list = data_list
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]


def main():
    # -------- Paths (repo-root relative) --------
    DIR_DATA = REPO_ROOT / "data"
    DIR_OUT  = REPO_ROOT / "GEOM_1D_graph_datasets"

    DIR_OUT.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("BUILDING CARBON-13 NMR DATASET & SAVING FIXED SPLITS")
    print("=" * 80)

    # -------- Build dataset once (CNMR) --------
    cfg = MatchedSplitConfig(
        graph_npz_path=str(DIR_DATA / "npz_data" / "dataset_graph_C_with_3d_matched.npz"),
        match_csv_path=str(DIR_DATA / "C_graph_geom_matches.csv"),
        conformer_folder=str(DIR_DATA / "conformer_pickles_C_matched"),
        task_name="C",
        verbose=True,
    )

    data_list = build_dataset(cfg)
    dataset   = SimpleDataset(data_list)
    n = len(dataset)
    print(f"Total samples: {n}")

    # -------- Deterministic split (frozen) --------
    seed_for_split = 2025
    gen = torch.Generator().manual_seed(seed_for_split)

    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val

    perm = torch.randperm(n, generator=gen).tolist()
    train_idx = np.array(perm[:n_train], dtype=np.int64)
    val_idx   = np.array(perm[n_train:n_train + n_val], dtype=np.int64)
    test_idx  = np.array(perm[n_train + n_val:], dtype=np.int64)

    # -------- Save artifacts (repo-root/GEOM_graph_datasets) --------
    split_path = DIR_OUT / "C_split_indices.npz"
    np.savez(split_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    print(f"Saved split indices -> {split_path}")
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    ds_path = DIR_OUT / "C_dataset_list.pt"
    torch.save(data_list, ds_path)
    print(f"Saved dataset list (order-frozen) -> {ds_path}")

    # -------- Save metadata --------
    meta = {
        "task": "C",
        "n_total": int(n),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "dataset_path": str(ds_path),
        "split_path": str(split_path),
        "seed_for_split": seed_for_split,
        "repo_root": str(REPO_ROOT),
        "data_root": str(DIR_DATA),
    }
    meta_path = DIR_OUT / "C_dataset_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta -> {meta_path}")


if __name__ == "__main__":
    main()
