import os
import json
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data, DataLoader
from datasets import load_dataset
from rdkit import Chem


# ======= CONFIG (edit paths if needed) =======
CSV_PATH          = "./2DNMRGym_graph_index_map_with_geom_matched.csv"
MAPPING_REPORT    = "./geom_mapping_report.csv"
DEFAULT_BATCHSIZE = 32


def _canon(s: str):
    if not isinstance(s, str): return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def read_good_matches(mapping_report_csv: str) -> pd.DataFrame:
    """
    Keep only rows where we have a valid, consistent atom mapping:
      status in {"same_order", "permuted"} (and NOT "*_BUT_inconsistent"),
      perm_json is a non-empty path that exists.
    """
    rep = pd.read_csv(mapping_report_csv, dtype={"filename": str})
    # rep["filename"] = rep["filename"].astype(str).str.strip()

    good_status = {"same_order", "permuted"}
    rep["is_good"] = rep["status"].isin(good_status)
    rep["perm_json"] = rep["perm_json"].fillna("")
    rep["has_json"] = rep["perm_json"].apply(lambda p: isinstance(p, str) and len(p) > 0 and os.path.exists(p))

    good = rep[(rep["is_good"]) & (rep["has_json"])].copy()
    return good[["filename", "geom_smiles", "perm_json", "n_atoms", "status"]]


def load_hsQC_graphs_filtered(
    csv_path: str,
    good_map_df: pd.DataFrame
) -> List[Data]:
    """
    Load all graphs from 'siriusxiao/2DNMRGym' (train+eval), filter to filenames in good_map_df,
    and attach CSV SMILES + perm_json + geom_smiles to each Data.
    """
    # Read CSV to bring SMILES by filename
    meta = pd.read_csv(csv_path, dtype={"filename": str})
    meta["filename"] = meta["filename"].astype(str).str.strip()

    # Left-join SMILES onto the good set to ensure we have graph SMILES per filename
    good = (good_map_df
            .merge(meta[["filename", "SMILES", "geom_smiles"]], on="filename", how="left", suffixes=("", "_csv")))
    # Prefer CSV SMILES column if present
    good["SMILES"] = good["SMILES"].astype(str)

    keep_fns = set(good["filename"].astype(str))
    fn2smi   = dict(zip(good["filename"].astype(str), good["SMILES"].astype(str)))
    fn2perm  = dict(zip(good["filename"].astype(str), good["perm_json"].astype(str)))
    fn2geom  = dict(zip(good["filename"].astype(str), good["geom_smiles"].astype(str)))

    ds_train = list(load_dataset("siriusxiao/2DNMRGym", split="train"))
    ds_eval  = list(load_dataset("siriusxiao/2DNMRGym", split="eval"))
    raw = ds_train + ds_eval

    graphs: List[Data] = []
    miss = 0

    for s in raw:
        fname = str(s["filename"]).strip()
        if fname not in keep_fns:
            continue

        g = s["graph_data"]
        data = Data(
            x=torch.tensor(g["x"], dtype=torch.float),
            edge_index=torch.tensor(g["edge_index"], dtype=torch.long),
            edge_attr=(torch.tensor(g["edge_attr"], dtype=torch.long)
                       if g.get("edge_attr") is not None else None),
            pos=(torch.tensor(g["pos"], dtype=torch.float)
                 if g.get("pos") is not None else None),
            solvent_class=torch.tensor([g["solvent_class"]], dtype=torch.long),
            cnmr=torch.tensor(s["c_peaks"], dtype=torch.float),
            hnmr=torch.tensor(s["h_peaks"], dtype=torch.float),
            filename=fname,
        )

        # Attach canonical graph smiles (from CSV) for mapping
        smi = fn2smi.get(fname, None)
        if smi is not None and isinstance(smi, str):
            data.smiles = _canon(smi)

        # Attach mapping JSON path (rdH->graphH mapping, and inverse)
        data.perm_json = fn2perm.get(fname, "")

        # Attach geom_smiles for convenience (helps locate pickle for conformers)
        data.geom_smiles = fn2geom.get(fname, "")

        # Sanity check: ensure mapping json exists
        if not (isinstance(data.perm_json, str) and os.path.exists(data.perm_json)):
            miss += 1
            continue

        graphs.append(data)

    if miss > 0:
        print(f"[HSQC filter] Skipped {miss} samples lacking a present perm_json on disk.")
    print(f"[HSQC filter] Loaded {len(graphs)} graphs with valid mappings.")
    return graphs


def split_80_10_10(graphs: List[Data], seed: int = 42) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Deterministic 80/10/10 split by filename (shuffle by seed).
    """
    rng = random.Random(seed)
    idx = list(range(len(graphs)))
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(round(0.8 * n))
    n_val   = int(round(0.1 * n))
    n_test  = n - n_train - n_val

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    train_set = [graphs[i] for i in train_idx]
    val_set   = [graphs[i] for i in val_idx]
    test_set  = [graphs[i] for i in test_idx]

    print(f"[HSQC split] train/val/test = {len(train_set)}/{len(val_set)}/{len(test_set)} (total={n})")
    return train_set, val_set, test_set


def make_dataloaders(
    train_set: List[Data],
    val_set: List[Data],
    test_set: List[Data],
    batch_size: int = DEFAULT_BATCHSIZE
) -> Dict[str, DataLoader]:
    loaders = {
        "train": DataLoader(train_set, shuffle=True,  batch_size=batch_size),
        "val":   DataLoader(val_set,   shuffle=False, batch_size=batch_size),
        "test":  DataLoader(test_set,  shuffle=False, batch_size=batch_size),
    }
    return loaders


# ======= Convenience one-call pipeline =======
def build_hsqc_matched_dataloaders(
    csv_path: str = CSV_PATH,
    mapping_report_csv: str = MAPPING_REPORT,
    batch_size: int = DEFAULT_BATCHSIZE,
    seed: int = 42
) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    """
    Returns:
      dataloaders: dict(train/val/test -> DataLoader)
      stats: counts for transparency
    """
    good = read_good_matches(mapping_report_csv)
    graphs = load_hsQC_graphs_filtered(csv_path, good)
    train_set, val_set, test_set = split_80_10_10(graphs, seed=seed)
    loaders = make_dataloaders(train_set, val_set, test_set, batch_size=batch_size)
    stats = {
        "total_good_rows": int(len(good)),
        "graphs_loaded":   int(len(graphs)),
        "train":           int(len(train_set)),
        "val":             int(len(val_set)),
        "test":            int(len(test_set)),
    }
    return loaders, stats


if __name__ == "__main__":
    loaders, stats = build_hsqc_matched_dataloaders(
        csv_path=CSV_PATH,
        mapping_report_csv=MAPPING_REPORT,
        batch_size=32,
        seed=42,
    )
    print("[HSQC] dataloader stats:", stats)