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

# Import central configuration
import C_NMR_config as config

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


# ---------------- Helpers for ensemble ----------------
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


def match_conformer_to_graph_coords(graph_smiles, conf_mol):
    if conf_mol is None or conf_mol.GetNumConformers() == 0:
        return None
    try:
        graph_mol = Chem.MolFromSmiles(graph_smiles)
        if graph_mol is None:
            return None
        conf_no_h = Chem.RemoveHs(conf_mol, sanitize=False)
        graph_no_h, graph_order = canonical_nonH_order(graph_mol)
        conf_no_h2, conf_order = canonical_nonH_order(conf_no_h)
        if conf_no_h2.GetNumAtoms() != graph_no_h.GetNumAtoms():
            return None

        inv_conf = {c: i for i, c in enumerate(conf_order)}
        perm = np.array([inv_conf[g] for g in graph_order])

        non_h_idx = [i for i, a in enumerate(conf_mol.GetAtoms()) if a.GetAtomicNum() != 1]
        all_xyz = conf_mol.GetConformer(0).GetPositions()
        conf_coords = all_xyz[non_h_idx]
        graph_coords = conf_coords[perm]
        return graph_coords.astype(np.float32)
    except Exception:
        return None


@torch.no_grad()
def evaluate_raw(model, loader, device):
    """Evaluate model (raw ppm values)."""
    model.eval()
    pred, true = evaluate(model, loader, device)
    return pred, true


@torch.no_grad()
def evaluate_ensemble_with_atoms(model_2d, model_3d, test_set, conformer_folder, device, top_n=30):
    """Multi-conformer ensemble evaluation (raw ppm values)."""

    def sanitize_filename(s):
        return s.replace('/', '_').replace('\\', '_').replace(':', '_')

    model_2d.eval()
    model_3d.eval()

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
            pos_np = match_conformer_to_graph_coords(smiles, rd_mol)
            if pos_np is None:
                continue
            d = data.clone()
            d.pos = torch.tensor(pos_np, dtype=torch.float32)
            d = d.to(device)
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

        # Annotate atoms with metadata
        try:
            mol_rd = Chem.MolFromSmiles(smiles)
            mol_no_h, graph_order = canonical_nonH_order(mol_rd)
            nonH_atoms = [mol_no_h.GetAtomWithIdx(i) for i in graph_order]
            carbon_atoms = [(i, a) for i, a in enumerate(nonH_atoms) if a.GetAtomicNum() == 6]
            use_meta = (len(carbon_atoms) == n_masked)
        except Exception:
            use_meta = False

        # Store ALL error metrics
        for a_idx in range(n_masked):
            # Compute all error metrics
            err_2d = abs(true_vals[a_idx] - pred_2d[a_idx])
            err_3d_single = abs(true_vals[a_idx] - pred_3d_single[a_idx])
            err_3d_ensemble = abs(true_vals[a_idx] - pred_3d_ensemble[a_idx])

            # Absolute improvement (3D vs 2D)
            abs_improvement = err_2d - err_3d_ensemble

            # Relative improvement (percentage)
            rel_improvement = ((err_2d - err_3d_ensemble) / err_2d * 100) if err_2d > 1e-6 else 0.0

            # Get atom metadata
            atom_label = f"C[{a_idx}]"
            hybrid_str = "UNK"
            if use_meta:
                _, atom = carbon_atoms[a_idx]
                atom_label = f"{atom.GetSymbol()}[{a_idx}]"
                hybrid_str = hybrid_to_str(atom.GetHybridization())

            atom_rows.append({
                "smiles": smiles,
                "atom": atom_label,
                "hybrid": hybrid_str,
                "true": float(true_vals[a_idx]),
                "pred_2d": float(pred_2d[a_idx]),
                "pred_3d_single": float(pred_3d_single[a_idx]),
                "pred_3d_ensemble": float(pred_3d_ensemble[a_idx]),
                "error_2d": float(err_2d),
                "error_3d_single": float(err_3d_single),
                "error_3d_ensemble": float(err_3d_ensemble),
                "abs_improvement": float(abs_improvement),
                "rel_improvement": float(rel_improvement),
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

    df_atoms = pd.DataFrame(atom_rows)
    df_mols = pd.DataFrame(mol_rows)
    return mae_2d_global, mae_3s_global, mae_3e_global, df_atoms, df_mols


def print_top_improved_atoms(df_atoms, top_n=30):
    """
    Print top N atoms ranked by both:
    1) Strong improvement vs 2D baseline
    2) High accuracy (close to true value)

    Composite score = improvement - penalty_for_error
    """
    if df_atoms.empty:
        print("\n(No per-atom data recorded – check conformer files/matching.)")
        return

    # Compute composite score: balance improvement AND final accuracy
    # Higher score = better (large improvement + low final error)
    df_atoms = df_atoms.copy()
    df_atoms['composite_score'] = df_atoms['abs_improvement'] - 0.85 * df_atoms['error_3d_ensemble']

    # Sort by composite score (best first)
    df_sorted = df_atoms.sort_values('composite_score', ascending=False)

    # Print header
    print(f"\n{'=' * 130}")
    print(f"TOP {top_n} IMPROVED & ACCURATE ATOMS — ¹³C NMR")
    print(f"{'=' * 130}")
    print(f"Ranking criteria: (1) Strong improvement vs 2D, (2) High final accuracy")
    print(f"{'=' * 130}")

    print(f"{'Rank':<5} {'SMILES':<40} {'Atom':<10} {'Hyb':<5} "
          f"{'True':>8} {'2D':>8} {'3D Ens':>8} {'Improv':>8} {'3D Err':>8}")
    print("-" * 130)

    for rank, row in enumerate(df_sorted.head(top_n).itertuples(index=False), 1):
        smiles_short = row.smiles[:37] + "..." if len(row.smiles) > 40 else row.smiles

        # Format improvement with sign
        improvement = row.abs_improvement
        improvement_str = f"{improvement:>+8.3f}"

        print(f"{rank:<5} {smiles_short:<40} {row.atom:<10} {row.hybrid:<5} "
              f"{row.true:>8.2f} {row.pred_2d:>8.2f} {row.pred_3d_ensemble:>8.2f} "
              f"{improvement_str} {row.error_3d_ensemble:>8.3f}")

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

    # Print configuration
    print(f"\n{'=' * 60}")
    print(f"TESTING CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"  Pretrained models: {'ENABLED' if config.IF_PRETRAIN else 'DISABLED'}")
    print(f"  Seeds: {config.SEEDS}")
    print(f"{'=' * 60}\n")

    if not (config.DATASET_PT.exists() and config.SPLIT_NPZ.exists()):
        raise FileNotFoundError(
            f"Missing dataset/splits.\n  {config.DATASET_PT}\n  {config.SPLIT_NPZ}\nRun make_splits_C.py first.")
    if not config.CONF_DIR.exists():
        print(
            f"Warning: Conformer folder not found: {config.CONF_DIR} — ensemble will be skipped where files are missing.")

    # Load dataset
    data_list = torch.load(str(config.DATASET_PT), map_location="cpu")
    dataset = SimpleDataset(data_list)
    split = np.load(str(config.SPLIT_NPZ))
    test_idx = split["test_idx"].tolist()
    test_set = Subset(dataset, test_idx)

    # Simple test loader
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Test size: {len(test_set)}")

    # Aggregation structure depends on IF_PRETRAIN
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
        print(f"\n{'=' * 80}\nSEED {SEED} — loading checkpoints and evaluating (CNMR)\n{'=' * 80}")

        path_2d = config.DIR_MODELS / f"C_2D_baseline_seed{SEED}.pt"
        path_3d = config.DIR_MODELS / f"C_3D_single_seed{SEED}.pt"
        path_3d_pt = config.DIR_MODELS / f"C_3D_single_pretrained_seed{SEED}.pt"

        # Store per-seed results for table
        seed_results = {}

        # ========== 2D BASELINE ==========
        if not path_2d.exists():
            print(f"Warning: Missing 2D checkpoint: {path_2d}")
            mae_2d = np.nan
            m2d = None
        else:
            m2d = build_2d_on(torch.device("cpu"))
            m2d = safe_load_weights(m2d, path_2d, device)

            p2d, t2d = evaluate_raw(m2d, test_loader, device)
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

            p3d, t3d = evaluate_raw(m3d, test_loader, device)
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

                p3d_pt, t3d_pt = evaluate_raw(m3d_pt, test_loader, device)
                mae_3d_pt_single = mean_absolute_error(t3d_pt, p3d_pt)
                print(f"[Seed {SEED}] 3D(pretrained single) Test MAE: {mae_3d_pt_single:.4f} ppm")
                agg["3D_pretrained_single"].append(mae_3d_pt_single)
                seed_results["3D pretrained (single)"] = mae_3d_pt_single

            # 3D pretrained ensemble
            if (m2d is not None) and (m3d_pt is not None) and config.CONF_DIR.exists():
                g2, g3s, ge, df_atoms, df_mols = evaluate_ensemble_with_atoms(
                    m2d, m3d_pt, test_set, config.CONF_DIR, device, top_n=config.TOP_N_ATOMS
                )
                print(f"[Seed {SEED}] 3D Pretrained Ensemble (global atom-wise) MAE: {ge:.4f} ppm")
                agg["3D_pretrained_ensemble"].append(ge)
                seed_results["3D pretrained (ensemble)"] = ge
                print_top_improved_atoms(df_atoms, top_n=config.TOP_N_ATOMS)
            else:
                print("(Info) Pretrained ensemble skipped for this seed (missing models or conformers).")

        # ========== NON-PRETRAINED ENSEMBLE PATH ==========
        else:
            if (m2d is not None) and (m3d is not None) and config.CONF_DIR.exists():
                g2, g3s, ge, df_atoms, df_mols = evaluate_ensemble_with_atoms(
                    m2d, m3d, test_set, config.CONF_DIR, device, top_n=config.TOP_N_ATOMS
                )
                print(f"[Seed {SEED}] 3D Ensemble (global atom-wise) MAE: {ge:.4f} ppm")
                agg["3D_ensemble"].append(ge)
                seed_results["3D ensemble"] = ge
                print_top_improved_atoms(df_atoms, top_n=config.TOP_N_ATOMS)
            else:
                print("(Info) Ensemble skipped for this seed (missing models or conformers).")

        # ========== PER-SEED SUMMARY TABLE ==========
        print(f"\n{'=' * 80}")
        print(f"SEED {SEED} SUMMARY")
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

    # ========== ACROSS-SEEDS SUMMARY ==========
    def m_s(arr):
        arr = np.array(arr, dtype=float)
        if len(arr) == 0:
            return np.nan, np.nan, 0
        return float(arr.mean()), float(arr.std(ddof=0)), len(arr)

    print(f"\n{'=' * 80}")
    print(f"ACROSS-SEEDS SUMMARY (CNMR) - Average over {len(config.SEEDS)} seeds")
    print(f"{'=' * 80}")

    # Compute baseline average for improvement calculation
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

    print(f"{'=' * 80}")
    print(f"Configuration: Pretrain={'ON' if config.IF_PRETRAIN else 'OFF'}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()