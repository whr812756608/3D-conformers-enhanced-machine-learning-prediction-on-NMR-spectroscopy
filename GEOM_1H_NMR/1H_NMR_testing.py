# 1H_NMR_testing.py - ¹H NMR prediction testing (follows C NMR workflow)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys, platform, pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, Subset
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error
import warnings
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import H_NMR_config as config

# ---------------- Project imports ----------------
from GEOM_models.Model_2D_3D_hybrid import NMR2DMPNN, NMR3DNet, evaluate

# ---------------- Device selection ----------------
USE_CPU = os.getenv("FORCE_CPU", "0") == "1"
IS_DARWIN = (platform.system() == "Darwin")
if IS_DARWIN:
    device = torch.device("cpu")
else:
    device = torch.device("cpu" if USE_CPU else ("cuda" if torch.cuda.is_available() else "cpu"))

# Determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------- Dataset wrapper ----------------
class SimpleDataset(TorchDataset):
    def __init__(self, data_list): self.data_list = data_list

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx): return self.data_list[idx]


def build_2d_on(device):
    return NMR2DMPNN(**config.MODEL_2D_CONFIG).to(device)


def build_3d_on(device):
    return NMR3DNet(**config.MODEL_3D_CONFIG).to(device)


# ---------------- Helpers for ¹H-specific ensemble ----------------
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType


def hybrid_to_str(hy):
    if hy == HybridizationType.SP3: return "SP3"
    if hy == HybridizationType.SP2: return "SP2"
    if hy == HybridizationType.SP:  return "SP"
    return "UNK"


def canonical_nonH_order(mol):
    mol_no_h = Chem.RemoveHs(mol, sanitize=False)
    ranks = list(Chem.CanonicalRankAtoms(mol_no_h, includeChirality=True))
    return mol_no_h, sorted(range(len(ranks)), key=lambda i: ranks[i])


def group_label(sym, nH, hyb, aromatic):
    """Basic, readable grouping for ¹H tables"""
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
            return "Vinylic H"
        # SP3: CH3/CH2/CH labels
        return "CH3" if nH == 3 else "CH2" if nH == 2 else "CH" if nH == 1 else "C"
    return sym


@torch.no_grad()
def evaluate_ensemble_with_atoms(model_2d, model_3d, test_set, conformer_folder, device,top_n=30):
    """
    Multi-conformer ensemble evaluation for ¹H NMR
    H-specific: Labels are attached to heavy atoms (site_idx refers to heavy atom index)
    """

    def sanitize_filename(s):
        return s.replace('/', '_').replace('\\', '_').replace(':', '_')

    model_2d.eval()
    model_3d.eval()
    conformer_folder = conformer_folder

    atom_rows, mol_rows = [], []
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

        # 2D base pred
        base = data.clone().to(device)
        base.batch = torch.zeros(base.x.size(0), dtype=torch.long, device=device)
        pred_2d = model_2d(base).detach().cpu().numpy()

        # 3D per conformer
        conf_preds, conf_w = [], []
        for conf in conformers:
            rd_mol = conf.get("rd_mol")
            w = float(conf.get("boltzmannweight", 0.0))
            if rd_mol is None or w == 0.0:
                continue
            if not hasattr(data, "pos") or data.pos is None:
                continue

            d = data.clone().to(device)
            d.batch = torch.zeros(d.x.size(0), dtype=torch.long, device=device)
            conf_preds.append(model_3d(d).detach().cpu().numpy())
            conf_w.append(w)

        if not conf_preds:
            continue

        conf_preds = np.array(conf_preds)
        conf_w = np.array(conf_w, dtype=np.float64).reshape(-1, 1)
        conf_w = conf_w / conf_w.sum()

        pred_3d_ensemble = (conf_preds * conf_w).sum(axis=0)
        pred_3d_single = conf_preds[0]
        true_vals = data.y[data.mask].cpu().numpy()


        n_masked = len(true_vals)

        all_true.append(true_vals)
        all_p2d.append(pred_2d)
        all_p3s.append(pred_3d_single)
        all_p3e.append(pred_3d_ensemble)

        mol_rows.append({
            "smiles": smiles,
            "n_atoms": n_masked,
            "n_conformers": conf_preds.shape[0],
            "mae_2d": float(np.mean(np.abs(true_vals - pred_2d))),
            "mae_3d_single": float(np.mean(np.abs(true_vals - pred_3d_single))),
            "mae_3d_ensemble": float(np.mean(np.abs(true_vals - pred_3d_ensemble))),
        })

        # Per-atom (only keep improved vs 2D)
        for a_idx in range(n_masked):
            err2d = abs(true_vals[a_idx] - pred_2d[a_idx])
            err3e = abs(true_vals[a_idx] - pred_3d_ensemble[a_idx])
            gain = err2d - err3e
            if gain <= 0:
                continue
            atom_rows.append({
                "smiles": smiles,
                "site_idx": int(a_idx),  # H NMR: index of heavy atom
                "true": float(true_vals[a_idx]),
                "pred_2d": float(pred_2d[a_idx]),
                "pred_3d_ensemble": float(pred_3d_ensemble[a_idx]),
                "gain": float(gain),
            })

    if len(all_true) == 0:
        return np.nan, np.nan, np.nan, pd.DataFrame(atom_rows), pd.DataFrame(mol_rows)

    T = np.concatenate(all_true, axis=0)
    P2 = np.concatenate(all_p2d, axis=0)
    P3 = np.concatenate(all_p3s, axis=0)
    PE = np.concatenate(all_p3e, axis=0)

    mae_2d_global = mean_absolute_error(T, P2)
    mae_3s_global = mean_absolute_error(T, P3)
    mae_3e_global = mean_absolute_error(T, PE)

    df_atoms = pd.DataFrame(atom_rows).sort_values("gain", ascending=False).reset_index(drop=True)
    df_mols = pd.DataFrame(mol_rows)
    return mae_2d_global, mae_3s_global, mae_3e_global, df_atoms, df_mols


def print_atom_improvements(df_atoms, top_n=30):
    """
    Pretty-print the top-N improved proton sites (¹H NMR).
    Enriches with RDKit metadata (CH3/CH2/CH/OH/NH, hybridization, ring).
    """
    if df_atoms.empty:
        print("\n(No per-atom improvements recorded – check conformer files/matching.)")
        return

    # Helpers
    def hyb_str(hy):
        return "SP3" if hy == HybridizationType.SP3 else \
            "SP2" if hy == HybridizationType.SP2 else \
                "SP" if hy == HybridizationType.SP else "UNK"

    # Build enriched view
    rows = []
    for row in df_atoms.head(top_n).itertuples(index=False):
        smiles = row.smiles
        idx = int(getattr(row, "site_idx", -1))

        # Defaults
        site = f"H@?[{idx}]"
        sym = "?"
        nH = 0
        hyb = "UNK"
        ring = "NA"
        group = "UNK"

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol_no_h, order = canonical_nonH_order(mol)
                if 0 <= idx < len(order):
                    heavy_idx = order[idx]
                    mol_H = Chem.AddHs(mol, addCoords=False)
                    heavy = mol_H.GetAtomWithIdx(heavy_idx)

                    sym = heavy.GetSymbol()
                    hyb = hyb_str(heavy.GetHybridization())
                    ring = "Yes" if heavy.IsInRing() else "No"
                    nH = sum(1 for nb in heavy.GetNeighbors() if nb.GetAtomicNum() == 1)

                    group = group_label(sym, nH, hyb, heavy.GetIsAromatic())
                    site = f"H@{sym}[{idx}]"
        except Exception:
            pass

        rows.append({
            "smiles": smiles,
            "site": site,
            "group": group,
            "nH": nH,
            "hybrid": hyb,
            "ring": ring,
            "true": float(getattr(row, 'true', float('nan'))),
            "pred_2d": float(getattr(row, 'pred_2d', float('nan'))),
            "pred_3d_ensemble": float(getattr(row, 'pred_3d_ensemble', float('nan'))),
            "gain": float(getattr(row, 'gain', float('nan'))),
        })

    view = pd.DataFrame(rows)

    # Pretty print
    print("\n" + "=" * 128)
    print(f"TOP {len(view)} IMPROVED PROTON SITES (3D Ensemble vs 2D) – ¹H NMR")
    print("=" * 128)
    print(f"{'Rank':<5} {'SMILES':<45} {'Site':<14} {'Group':<10} {'nH':>3} "
          f"{'Hybrid':<6} {'Ring':<4} {'True':>8} {'2D':>8} {'3D Ens':>9} {'Gain':>9}")
    print("-" * 128)

    for r, row in enumerate(view.itertuples(index=False), 1):
        smiles_short = row.smiles[:42] + "..." if len(row.smiles) > 45 else row.smiles
        print(f"{r:<5} {smiles_short:<45} {row.site:<14} {row.group:<10} {row.nH:>3} "
              f"{row.hybrid:<6} {row.ring:<4} {row.true:>8.2f} {row.pred_2d:>8.2f} "
              f"{row.pred_3d_ensemble:>9.2f} {row.gain:>+9.3f}")

    print("\nNote: Site = proton attached to heavy atom; 'nH' = number of protons on that heavy atom")


def safe_load_weights(model, ckpt_path, device):
    """Load weights safely."""
    state = torch.load(str(ckpt_path), map_location="cpu")

    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'], strict=False)
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  [warn] missing={len(missing)}, unexpected={len(unexpected)}")

    model.to(device)
    model.eval()
    return model


def main():
    print(f"Using device: {device}")
    print(f"Repo root: {config.REPO_ROOT}")

    print(f"\n{'=' * 60}")
    print(f"¹H NMR TESTING CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"  Pretrained models: {'ENABLED' if config.IF_PRETRAIN else 'DISABLED'}")
    print(f"  Seeds: {config.SEEDS}")
    print(f"{'=' * 60}\n")

    if not (config.DATASET_PT.exists() and config.SPLIT_NPZ.exists()):
        raise FileNotFoundError(
            f"Missing dataset/splits.\n  {config.DATASET_PT}\n  {config.SPLIT_NPZ}\nRun make_splits_H.py first.")
    if not config.CONF_DIR.exists():
        print(
            f"Warning: Conformer folder not found: {config.CONF_DIR} – ensemble will be skipped where files are missing.")

    # Load dataset
    data_list = torch.load(str(config.DATASET_PT), map_location="cpu")
    dataset = SimpleDataset(data_list)
    split = np.load(str(config.SPLIT_NPZ))
    test_idx = split["test_idx"].tolist()
    test_set = Subset(dataset, test_idx)

    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Test size: {len(test_set)}")

    # Aggregation structure
    if config.IF_PRETRAIN:
        agg = {
            "2D": [],
            "3D_single": [],
            "3D_pretrained_single": [],
            "3D_pretrained_ensemble": []
        }
    else:
        agg = {
            "2D": [],
            "3D_single": [],
            "3D_ensemble": []
        }

    for SEED in config.SEEDS:
        print(f"\n{'=' * 80}\nSEED {SEED} – loading checkpoints and evaluating (1H NMR)\n{'=' * 80}")

        path_2d = config.DIR_MODELS / f"H_2D_baseline_seed{SEED}.pt"
        path_3d = config.DIR_MODELS / f"H_3D_single_seed{SEED}.pt"
        path_3d_pt = config.DIR_MODELS / f"H_3D_single_pretrained_seed{SEED}.pt"

        seed_results = {}

        # ========== 2D BASELINE ==========
        if not path_2d.exists():
            print(f"Warning: Missing 2D checkpoint: {path_2d}")
            mae_2d = np.nan
            m2d = None
        else:
            m2d = build_2d_on(torch.device("cpu"))
            m2d = safe_load_weights(m2d, path_2d, device)

            p2d, t2d = evaluate(m2d, test_loader, device)
            mae_2d = mean_absolute_error(t2d, p2d)
            print(f"[Seed {SEED}] 2D Test MAE: {mae_2d:.4f} ppm")
            agg["2D"].append(mae_2d)
            seed_results["2D baseline"] = mae_2d

        # ========== 3D SINGLE (from scratch) ==========
        if not path_3d.exists():
            print(f"Warning: Missing 3D(single) checkpoint: {path_3d}")
            mae_3d_single = np.nan
            m3d = None
        else:
            m3d = build_3d_on(torch.device("cpu"))
            m3d = safe_load_weights(m3d, path_3d, device)
            p3d, t3d = evaluate(m3d, test_loader, device)
            mae_3d_single = mean_absolute_error(t3d, p3d)
            print(f"[Seed {SEED}] 3D(single) Test MAE: {mae_3d_single:.4f} ppm")
            agg["3D_single"].append(mae_3d_single)
            seed_results["3D single"] = mae_3d_single

        # ========== PRETRAINED PATH ==========
        if config.IF_PRETRAIN:
            if not path_3d_pt.exists():
                print(f"Warning: Missing 3D(pretrained) checkpoint: {path_3d_pt}")
                mae_3d_pt_single = np.nan
                m3d_pt = None
            else:
                m3d_pt = build_3d_on(torch.device("cpu"))
                m3d_pt = safe_load_weights(m3d_pt, path_3d_pt, device)
                p3d_pt, t3d_pt = evaluate(m3d_pt, test_loader, device)
                mae_3d_pt_single = mean_absolute_error(t3d_pt, p3d_pt)
                print(f"[Seed {SEED}] 3D(pretrained single) Test MAE: {mae_3d_pt_single:.4f} ppm")
                agg["3D_pretrained_single"].append(mae_3d_pt_single)
                seed_results["3D pretrained (single)"] = mae_3d_pt_single

            # 3D pretrained ensemble
            if (m2d is not None) and (m3d_pt is not None) and config.CONF_DIR.exists():
                g2, g3s, ge, df_atoms, df_mols = evaluate_ensemble_with_atoms(
                    m2d, m3d_pt, test_set, config.CONF_DIR, device, top_n=config.TOP_N_ATOMS,
                )
                print(f"[Seed {SEED}] 3D Pretrained Ensemble MAE: {ge:.4f} ppm")
                agg["3D_pretrained_ensemble"].append(ge)
                seed_results["3D pretrained (ensemble)"] = ge
                print_atom_improvements(df_atoms, top_n=config.TOP_N_ATOMS)

        # NON-PRETRAINED ENSEMBLE
        else:
            if (m2d is not None) and (m3d is not None) and config.CONF_DIR.exists():
                g2, g3s, ge, df_atoms, df_mols = evaluate_ensemble_with_atoms(
                    m2d, m3d_pt, test_set, config.CONF_DIR, device, top_n=config.TOP_N_ATOMS
                )
                print(f"[Seed {SEED}] 3D Ensemble MAE: {ge:.4f} ppm")
                agg["3D_ensemble"].append(ge)
                seed_results["3D ensemble"] = ge
                print_atom_improvements(df_atoms, top_n=config.TOP_N_ATOMS)

        # PER-SEED SUMMARY
        print(f"\n{'=' * 80}")
        print(f"SEED {SEED} SUMMARY (¹H NMR)")
        print(f"{'=' * 80}")
        print(f"{'Model':<35} {'MAE (ppm)':>12} {'vs 2D baseline':>18}")
        print("-" * 80)

        baseline_mae = seed_results.get("2D baseline", np.nan)
        for model_name, mae_val in seed_results.items():
            if not np.isnan(mae_val):
                if model_name == "2D baseline":
                    improvement_str = "baseline"
                else:
                    if not np.isnan(baseline_mae) and baseline_mae > 0:
                        improvement = ((baseline_mae - mae_val) / baseline_mae) * 100
                        improvement_str = f"{improvement:+.2f}%"
                    else:
                        improvement_str = "N/A"
                print(f"{model_name:<35} {mae_val:>12.4f} {improvement_str:>18}")
            else:
                print(f"{model_name:<35} {'N/A':>12} {'N/A':>18}")
        print("=" * 80)

    # ACROSS-SEEDS SUMMARY
    def m_s(arr):
        arr = np.array(arr, dtype=float)
        if len(arr) == 0:
            return np.nan, np.nan, 0
        return float(arr.mean()), float(arr.std(ddof=0)), len(arr)

    print(f"\n{'=' * 80}")
    print(f"ACROSS-SEEDS SUMMARY (¹H NMR) - Average over {len(config.SEEDS)} seeds")
    print(f"{'=' * 80}")

    baseline_mean, _, baseline_n = m_s(agg["2D"])

    if config.IF_PRETRAIN:
        comparisons = [
            ("2D", "2D baseline"),
            ("3D_single", "3D single (from scratch)"),
            ("3D_pretrained_single", "3D pretrained (single)"),
            ("3D_pretrained_ensemble", "3D pretrained (ensemble)")
        ]
    else:
        comparisons = [
            ("2D", "2D baseline"),
            ("3D_single", "3D single"),
            ("3D_ensemble", "3D ensemble")
        ]

    print(f"{'Model':<35} {'MAE (ppm)':>20} {'vs 2D baseline':>18}")
    print("-" * 80)

    for key, label in comparisons:
        mean_v, std_v, n_v = m_s(agg[key])
        if n_v > 0:
            mae_str = f"{mean_v:.4f} ± {std_v:.4f}"

            if key == "2D":
                improvement_str = "baseline"
            else:
                if not np.isnan(baseline_mean) and baseline_mean > 0 and not np.isnan(mean_v):
                    improvement = ((baseline_mean - mean_v) / baseline_mean) * 100
                    improvement_str = f"{improvement:+.2f}%"
                else:
                    improvement_str = "N/A"

            print(f"{label:<35} {mae_str:>20} {improvement_str:>18}")
        else:
            print(f"{label:<35} {'(no runs)':>20} {'N/A':>18}")



if __name__ == "__main__":
    main()