import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import os
import sys
import platform
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, Subset
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error
from GEOM_models.Model_2D_3D_hybrid import NMR2DMPNN, NMR3DNet, evaluate
# ---------------- Repo-root bootstrap ----------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------- Paths (repo-root relative) ----------
DIR_DATASETS = REPO_ROOT / "GEOM_1D_graph_datasets"
DIR_MODELS   = REPO_ROOT / "GEOM_models"
DIR_DATA     = REPO_ROOT / "data"

DATASET_PT   = DIR_DATASETS / "H_dataset_list.pt"
SPLIT_NPZ    = DIR_DATASETS / "H_split_indices.npz"
CONF_DIR     = DIR_DATA / "conformer_pickles_H_matched"  # ensemble source

# ---------------- Settings ----------------
SEEDS          = [1, 24, 66]
BATCH_SIZE     = 32
TOP_N_ATOMS    = 30
IS_DARWIN      = (platform.system() == "Darwin")
NUM_WORKERS    = 0   # macOS safer at 0
PIN_MEMORY     = False                  # CPU/macOS: keep False

# --- Helpers for 1H metadata ---
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType


# ---------------- Dataset wrapper ----------------
class SimpleDataset(TorchDataset):
    def __init__(self, data_list): self.data_list = data_list
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]


def build_2d(device):
    return NMR2DMPNN(
        node_feats=64, embed_feats=256,
        num_step_message_passing=5, num_step_set2set=3,
        hidden_feats=512, prob_dropout=0.1
    ).to(device)


def build_3d(device):
    return NMR3DNet(
        node_hidden=64, edge_hidden=128, n_interactions=4,
        use_3d=True, use_schnet=True, use_egnn=True, use_comenet=True, comenet_K=4
    ).to(device)


# ---------------- Multi-conformer ensemble + atom analysis ----------------
@torch.no_grad()
def evaluate_ensemble_with_atoms(model_2d, model_3d, test_set, conformer_folder, device, top_n=30):
    """Boltzmann-weighted ensemble for each molecule; keep only atoms improved vs 2D.
       This is the simple, previously working version (no extra RDKit metadata logic)."""
    from pathlib import Path
    import pickle
    import numpy as np
    import pandas as pd

    def sanitize_filename(s: str) -> str:
        return s.replace('/', '_').replace('\\', '_').replace(':', '_')

    model_2d.eval()
    model_3d.eval()
    conformer_folder = Path(conformer_folder)

    atom_rows = []
    mol_rows  = []
    all_true, all_p2d, all_p3s, all_p3e = [], [], [], []

    print(f"\nEvaluating {len(test_set)} test molecules...")

    for idx, data in enumerate(test_set, 1):
        if (idx % 50) == 0:
            print(f"  Processed {idx}/{len(test_set)}...")

        smiles = data.smiles
        pkl_files = list(conformer_folder.glob(f"{sanitize_filename(smiles)}*.pickle"))
        if not pkl_files:
            continue

        with open(pkl_files[0], "rb") as f:
            conf_data = pickle.load(f)
        conformers = conf_data.get("conformers", [])
        if not conformers:
            continue

        # 2D prediction on the base graph
        base = data.clone().to(device)
        base.batch = torch.zeros(base.x.size(0), dtype=torch.long, device=device)
        pred_2d = model_2d(base).cpu().numpy()

        # 3D predictions per conformer
        conf_preds, conf_w = [], []
        for conf in conformers:
            rd_mol = conf.get("rd_mol")
            w      = conf.get("boltzmannweight", 0.0)
            if rd_mol is None or w == 0.0:
                continue

            # positions already aligned to graph (your dataset builder ensures this)
            if not hasattr(data, "pos") or data.pos is None:
                continue

            d = data.clone().to(device)
            d.batch = torch.zeros(d.x.size(0), dtype=torch.long, device=device)
            pred = model_3d(d).cpu().numpy()
            conf_preds.append(pred)
            conf_w.append(w)

        if not conf_preds:
            continue

        conf_preds = np.array(conf_preds)                  # [K, N_masked]
        conf_w     = np.array(conf_w, dtype=np.float64)    # [K]
        conf_w     = (conf_w / conf_w.sum()).reshape(-1, 1)

        pred_3d_ensemble = (conf_preds * conf_w).sum(axis=0)
        pred_3d_single   = conf_preds[0]
        true_vals        = data.y[data.mask].cpu().numpy()
        n_masked         = len(true_vals)

        # accumulate globals
        all_true.append(true_vals)
        all_p2d.append(pred_2d)
        all_p3s.append(pred_3d_single)
        all_p3e.append(pred_3d_ensemble)

        # molecule-level MAE summary
        mol_rows.append({
            "smiles": smiles,
            "n_atoms": n_masked,
            "n_conformers": conf_preds.shape[0],
            "mae_2d": float(np.mean(np.abs(true_vals - pred_2d))),
            "mae_3d_single": float(np.mean(np.abs(true_vals - pred_3d_single))),
            "mae_3d_ensemble": float(np.mean(np.abs(true_vals - pred_3d_ensemble))),
        })

        # per-atom (only keep improved vs 2D)
        for a_idx in range(n_masked):
            err2d = abs(true_vals[a_idx] - pred_2d[a_idx])
            err3e = abs(true_vals[a_idx] - pred_3d_ensemble[a_idx])
            gain  = err2d - err3e
            if gain <= 0:
                continue
            atom_rows.append({
                "smiles": smiles,
                "atom_idx": int(a_idx),
                "true": float(true_vals[a_idx]),
                "pred_2d": float(pred_2d[a_idx]),
                "pred_3d_ensemble": float(pred_3d_ensemble[a_idx]),
                "gain": float(gain),
            })

    # global MAEs
    if len(all_true) == 0:
        df_atoms = pd.DataFrame(atom_rows)
        df_mols  = pd.DataFrame(mol_rows)
        return np.nan, np.nan, np.nan, df_atoms, df_mols

    T  = np.concatenate(all_true, axis=0)
    P2 = np.concatenate(all_p2d,   axis=0)
    P3 = np.concatenate(all_p3s,   axis=0)
    PE = np.concatenate(all_p3e,   axis=0)

    mae_2d_global = mean_absolute_error(T, P2)
    mae_3s_global = mean_absolute_error(T, P3)
    mae_3e_global = mean_absolute_error(T, PE)

    df_atoms = pd.DataFrame(atom_rows).sort_values("gain", ascending=False).reset_index(drop=True)
    df_mols  = pd.DataFrame(mol_rows)
    return mae_2d_global, mae_3s_global, mae_3e_global, df_atoms, df_mols


def print_atom_improvements(df_atoms, top_n=30):
    """
    Pretty-print the top-N improved proton sites (¹H).
    Works with your current simple evaluator output that has:
      - 'smiles'
      - 'site_idx' (or legacy 'atom_idx')  -> masked heavy-atom index the ¹H label is attached to
      - 'true', 'pred_2d', 'pred_3d_ensemble', 'gain'
    Enriches rows with RDKit-derived metadata (attached atom, group CH3/CH2/CH/OH/NH/..., hybrid, ring).
    """
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem.rdchem import HybridizationType

    if df_atoms.empty:
        print("\n(No per-atom improvements recorded — check conformer files/matching.)")
        return

    # --- helpers ---
    def hyb_str(hy):
        return "SP3" if hy == HybridizationType.SP3 else \
               "SP2" if hy == HybridizationType.SP2 else \
               "SP"  if hy == HybridizationType.SP  else "UNK"

    def canonical_nonH_order(mol):
        mol_no_h = Chem.RemoveHs(mol, sanitize=False)
        ranks = list(Chem.CanonicalRankAtoms(mol_no_h, includeChirality=True))
        order = sorted(range(len(ranks)), key=lambda i: ranks[i])
        return mol_no_h, order

    def group_label(sym, nH, hyb, aromatic):
        # Basic, readable grouping for ¹H tables
        if sym in ("O", "N", "S"):
            if nH >= 1:
                return f"{sym}H"  # OH / NH / SH
            return sym
        if sym == "C":
            if aromatic:
                return "Aromatic H"
            if hyb == "SP":
                return "Alkynyl H"
            if hyb == "SP2":
                return "Vinylic H"  # catches aldehydic as subset; keep it simple
            # SP3: CH3/CH2/CH labels
            return "CH3" if nH == 3 else "CH2" if nH == 2 else "CH" if nH == 1 else "C"
        return sym

    # ensure we have the needed columns
    idx_col = "site_idx" if "site_idx" in df_atoms.columns else \
              ("atom_idx" if "atom_idx" in df_atoms.columns else None)
    if idx_col is None:
        # fallback: just print the minimal, like your old table
        view = df_atoms.head(top_n)
        print("\n" + "="*100)
        print(f"TOP {len(view)} IMPROVED SITES (3D Ensemble vs 2D)")
        print("="*100)
        print(f"{'Rank':<5} {'SMILES':<45} {'Idx':>6} {'True':>8} {'2D':>8} {'3D Ens':>9} {'Gain':>9}")
        print("-"*100)
        for r, row in enumerate(view.itertuples(index=False), 1):
            smiles_short = row.smiles[:42] + "..." if len(row.smiles) > 45 else row.smiles
            idx = getattr(row, "atom_idx", -1)
            print(f"{r:<5} {smiles_short:<45} {idx:>6} "
                  f"{getattr(row, 'true', float('nan')):>8.2f} "
                  f"{getattr(row, 'pred_2d', float('nan')):>8.2f} "
                  f"{getattr(row, 'pred_3d_ensemble', float('nan')):>9.2f} "
                  f"{getattr(row, 'gain', float('nan')):>+9.3f}")
        print("\nNote: Idx = masked heavy-atom index the ¹H label is attached to.\n")
        return

    # Build an enriched view using RDKit
    rows = []
    for row in df_atoms.head(top_n).itertuples(index=False):
        smiles = row.smiles
        idx    = int(getattr(row, idx_col))
        # defaults
        site   = f"H@?[{idx}]"
        sym    = "?"
        nH     = 0
        hyb    = "UNK"
        ring   = "NA"
        group  = "UNK"

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # heavy-atom canonical order must match the graph’s convention
                mol_no_h, order = canonical_nonH_order(mol)
                if 0 <= idx < len(order):
                    heavy_idx = order[idx]
                    # Add explicit Hs (heavy atom indices are preserved)
                    mol_H = Chem.AddHs(mol, addCoords=False)
                    heavy = mol_H.GetAtomWithIdx(heavy_idx)

                    sym  = heavy.GetSymbol()
                    hyb  = hyb_str(heavy.GetHybridization())
                    ring = "Yes" if heavy.IsInRing() else "No"
                    nH   = sum(1 for nb in heavy.GetNeighbors() if nb.GetAtomicNum() == 1)

                    group = group_label(sym, nH, hyb, heavy.GetIsAromatic())
                    site  = f"H@{sym}[{idx}]"
        except Exception:
            pass

        rows.append({
            "smiles": smiles,
            "site": site,          # H@C[idx]
            "group": group,        # CH3/CH2/CH/OH/NH/...
            "nH": nH,              # number of H on that heavy atom (group size)
            "hybrid": hyb,
            "ring": ring,
            "true": float(getattr(row, 'true', float('nan'))),
            "pred_2d": float(getattr(row, 'pred_2d', float('nan'))),
            "pred_3d_ensemble": float(getattr(row, 'pred_3d_ensemble', float('nan'))),
            "gain": float(getattr(row, 'gain', float('nan'))),
        })

    view = pd.DataFrame(rows)

    # --- pretty print ---
    print("\n" + "="*128)
    print(f"TOP {len(view)} IMPROVED PROTON SITES (3D Ensemble vs 2D) — HNMR")
    print("="*128)
    print(f"{'Rank':<5} {'SMILES':<45} {'Site':<14} {'Group':<10} {'nH':>3} "
          f"{'Hybrid':<6} {'Ring':<4} {'True':>8} {'2D':>8} {'3D Ens':>9} {'Gain':>9}")
    print("-"*128)

    for r, row in enumerate(view.itertuples(index=False), 1):
        smiles_short = row.smiles[:42] + "..." if len(row.smiles) > 45 else row.smiles
        print(f"{r:<5} {smiles_short:<45} {row.site:<14} {row.group:<10} {row.nH:>3} "
              f"{row.hybrid:<6} {row.ring:<4} {row.true:>8.2f} {row.pred_2d:>8.2f} "
              f"{row.pred_3d_ensemble:>9.2f} {row.gain:>+9.3f}")

    print("\nNote: Site = proton attached to the heavy atom shown; "
          "‘nH’ is the number of protons on that heavy atom (e.g., 3 for CH3).")






def per_seed_table(seed, mae_2d, mae_3d_single, mae_3d_ens):
    print(f"\n{'-'*60}")
    print(f"SEED {seed} — CNMR test performance")
    print(f"{'-'*60}")
    print(f"{'Model':<22} {'Test MAE (ppm)':>14} {'Improvement vs 2D (%)':>24}")
    print("-" * 60)
    def imp(v):
        return ((mae_2d - v) / mae_2d * 100.0) if np.isfinite(mae_2d) and mae_2d > 0 else np.nan
    print(f"{'2D baseline':<22} {mae_2d:>14.4f} {0:>24.2f}")
    print(f"{'3D single conformer':<22} { (mae_3d_single if np.isfinite(mae_3d_single) else float('nan')):>14.4f} "
          f"{ (imp(mae_3d_single) if np.isfinite(mae_3d_single) else float('nan')):>24.2f}")
    print(f"{'3D multi-conformer':<22} { (mae_3d_ens if np.isfinite(mae_3d_ens) else float('nan')):>14.4f} "
          f"{ (imp(mae_3d_ens) if np.isfinite(mae_3d_ens) else float('nan')):>24.2f}")


def main():
    # macOS safer runtime (spawn only matters if you ever use workers>0)
    if IS_DARWIN:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)

    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Repo root: {REPO_ROOT}")

    # load frozen dataset/split
    if not (DATASET_PT.exists() and SPLIT_NPZ.exists()):
        raise FileNotFoundError(f"Missing dataset/splits.\n  {DATASET_PT}\n  {SPLIT_NPZ}\nRun make_splits_C.py first.")
    if not CONF_DIR.exists():
        print(f"⚠ Conformer folder not found: {CONF_DIR} — ensemble will be skipped where files are missing.")

    data_list = torch.load(str(DATASET_PT), map_location="cpu")
    dataset   = SimpleDataset(data_list)
    split     = np.load(str(SPLIT_NPZ))
    test_idx  = split["test_idx"].tolist()
    test_set  = Subset(dataset, test_idx)

    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=False
    )
    print(f"Test size: {len(test_set)}")

    # aggregate across seeds
    agg = {"2D": [], "3D_single": [], "3D_ensemble": []}
    rows_csv = [("seed", "model", "test_mae_ppm")]

    for SEED in SEEDS:
        print(f"\n{'='*80}\nSEED {SEED} — loading checkpoints and evaluating (CNMR)\n{'='*80}")
        path_2d = DIR_MODELS / f"H_2D_baseline_seed{SEED}.pt"
        path_3d = DIR_MODELS / f"H_3D_single_seed{SEED}.pt"

        # 2D
        if not path_2d.exists():
            print(f"⚠ Missing 2D checkpoint: {path_2d}")
            mae_2d = np.nan
            m2d = None
        else:
            m2d = build_2d(device)
            m2d.load_state_dict(torch.load(str(path_2d), map_location=device))
            m2d.eval()
            p2d, t2d = evaluate(m2d, test_loader, device)
            mae_2d = mean_absolute_error(t2d, p2d)
            print(f"[Seed {SEED}] 2D Test MAE: {mae_2d:.4f} ppm")
            agg["2D"].append(mae_2d)
            rows_csv.append((SEED, "2D", f"{mae_2d:.6f}"))

        # 3D single
        if not path_3d.exists():
            print(f"⚠ Missing 3D(single) checkpoint: {path_3d}")
            mae_3d_single = np.nan
            m3d = None
        else:
            m3d = build_3d(device)
            m3d.load_state_dict(torch.load(str(path_3d), map_location=device))
            m3d.eval()
            p3d, t3d = evaluate(m3d, test_loader, device)
            mae_3d_single = mean_absolute_error(t3d, p3d)
            print(f"[Seed {SEED}] 3D(single) Test MAE: {mae_3d_single:.4f} ppm")
            agg["3D_single"].append(mae_3d_single)
            rows_csv.append((SEED, "3D_single", f"{mae_3d_single:.6f}"))

        # 3D ensemble (needs both models + conformers)
        if (m2d is not None) and (m3d is not None) and CONF_DIR.exists():
            g2, g3s, ge, df_atoms, df_mols = evaluate_ensemble_with_atoms(
                m2d, m3d, test_set, CONF_DIR, device, top_n=TOP_N_ATOMS
            )
            mae_3d_ensemble = ge
            print(f"[Seed {SEED}] 3D Ensemble (global atom-wise) MAE: {mae_3d_ensemble:.4f} ppm")
            agg["3D_ensemble"].append(mae_3d_ensemble)
            rows_csv.append((SEED, "3D_ensemble", f"{mae_3d_ensemble:.6f}"))

            # per-seed table and atom improvements
            per_seed_table(SEED, mae_2d, mae_3d_single, mae_3d_ensemble)
            print_atom_improvements(df_atoms, top_n=TOP_N_ATOMS)
        else:
            print("(Info) Ensemble skipped for this seed (missing models or conformers).")
            per_seed_table(SEED, mae_2d, mae_3d_single, np.nan)

    # across-seed summary
    def m_s(arr):
        arr = np.array(arr, dtype=float)
        if len(arr) == 0:
            return np.nan, np.nan, 0
        return float(arr.mean()), float(arr.std(ddof=0)), len(arr)

    print(f"\n{'='*80}\nACROSS-SEEDS SUMMARY (CNMR)\n{'='*80}")
    for key, label in [("2D", "2D baseline"),
                       ("3D_single", "3D single conformer"),
                       ("3D_ensemble", "3D multi-conformer")]:
        mean_v, std_v, n_v = m_s(agg[key])
        if n_v > 0:
            print(f"{label:<24} mean={mean_v:>8.4f}  std={std_v:>8.4f}  n={n_v}")
        else:
            print(f"{label:<24} (no runs)")

    # # save per-seed CSV
    # out_csv = DIR_MODELS / "H_results_by_seed.csv"
    # pd.DataFrame(rows_csv).to_csv(out_csv, header=False, index=False)
    # print(f"\nSaved per-seed results -> {out_csv}")


if __name__ == "__main__":
    main()
