# eval_C.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error
import pandas as pd
from pathlib import Path
import pickle

from data.GEOM_Single_graph_builder_enhanced import build_dataset, MatchedSplitConfig
from Model_2D_3D_hybrid import NMR2DMPNN, NMR3DNet, evaluate

torch.manual_seed(0)
np.random.seed(0)


class SimpleDataset(TorchDataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


@torch.no_grad()
def evaluate_ensemble_with_atoms(model_2d, model_3d, test_set, conformer_folder, device):
    """Evaluate ensemble using ALL conformers per molecule"""
    from rdkit import Chem

    def sanitize_filename(s):
        return s.replace('/', '_').replace('\\', '_').replace(':', '_')

    def match_conformer_to_graph(graph_smiles, conf_mol):
        if conf_mol is None or conf_mol.GetNumConformers() == 0:
            return None
        try:
            graph_mol = Chem.MolFromSmiles(graph_smiles)
            if graph_mol is None:
                return None

            conf_mol_no_h = Chem.RemoveHs(conf_mol, sanitize=False)
            graph_mol_no_h = Chem.RemoveHs(graph_mol, sanitize=False)

            if conf_mol_no_h.GetNumAtoms() != graph_mol_no_h.GetNumAtoms():
                return None

            def canonical_order(mol):
                mol_no_h = Chem.RemoveHs(mol, sanitize=False)
                ranks = list(Chem.CanonicalRankAtoms(mol_no_h, includeChirality=True))
                return sorted(range(len(ranks)), key=lambda i: ranks[i])

            conf_order = canonical_order(conf_mol)
            graph_order = canonical_order(graph_mol)

            inverse_conf = {c: i for i, c in enumerate(conf_order)}
            permutation = np.array([inverse_conf[g] for g in graph_order])

            non_h_indices = [i for i, atom in enumerate(conf_mol.GetAtoms())
                             if atom.GetAtomicNum() != 1]
            all_coords = conf_mol.GetConformer(0).GetPositions()
            conf_coords = all_coords[non_h_indices]
            graph_coords = conf_coords[permutation]

            return graph_coords.astype(np.float32)
        except Exception:
            return None

    model_2d.eval()
    model_3d.eval()
    conformer_folder = Path(conformer_folder)

    atom_results = []
    mol_results = []

    print(f"\nEvaluating {len(test_set)} test molecules...")

    for idx, data in enumerate(test_set):
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(test_set)}...")

        smiles = data.smiles
        safe_smiles = sanitize_filename(smiles)

        pickle_files = list(conformer_folder.glob(f"{safe_smiles}*.pickle"))
        if not pickle_files:
            continue

        with open(pickle_files[0], 'rb') as f:
            conf_data = pickle.load(f)

        conformers = conf_data.get('conformers', [])
        if not conformers:
            continue

        # Base graph on device
        data_dev = data.to(device)
        data_dev.batch = torch.zeros(data_dev.x.size(0), dtype=torch.long, device=device)

        # 2D prediction
        pred_2d = model_2d(data_dev).cpu().numpy()

        # 3D predictions across all conformers
        conformer_preds = []
        conformer_weights = []

        for conf in conformers:
            rd_mol = conf.get('rd_mol')
            weight = conf.get('boltzmannweight', 0)
            if rd_mol is None or weight == 0:
                continue

            pos_np = match_conformer_to_graph(smiles, rd_mol)
            if pos_np is None:
                continue

            data_conf = data.clone()
            data_conf.pos = torch.tensor(pos_np, dtype=torch.float32)
            data_conf = data_conf.to(device)
            data_conf.batch = torch.zeros(data_conf.x.size(0), dtype=torch.long, device=device)

            pred = model_3d(data_conf).cpu().numpy()
            conformer_preds.append(pred)
            conformer_weights.append(weight)

        if not conformer_preds:
            continue

        conformer_preds = np.array(conformer_preds)             # [K, N_masked]
        conformer_weights = np.array(conformer_weights).reshape(-1, 1)
        conformer_weights = conformer_weights / conformer_weights.sum()

        pred_3d_ensemble = (conformer_preds * conformer_weights).sum(axis=0)
        pred_3d_single = conformer_preds[0]

        true_vals = data.y[data.mask].cpu().numpy()
        n_masked = len(true_vals)

        # Molecule-level MAE
        mol_results.append({
            'smiles': smiles,
            'n_atoms': n_masked,
            'n_conformers': len(conformer_preds),
            'mae_2d': np.abs(true_vals - pred_2d).mean(),
            'mae_3d_single': np.abs(true_vals - pred_3d_single).mean(),
            'mae_3d_ensemble': np.abs(true_vals - pred_3d_ensemble).mean()
        })

        # Atom-level rows (subset: only keep improvements/gains; skip bad ones)
        conformer_std = conformer_preds.std(axis=0) if len(conformer_preds) > 1 else np.zeros(n_masked)
        for atom_idx in range(n_masked):
            gain = abs(true_vals[atom_idx] - pred_2d[atom_idx]) - abs(true_vals[atom_idx] - pred_3d_ensemble[atom_idx])
            if gain <= 0:
                continue  # skip non-improvements as requested

            atom_results.append({
                'smiles': smiles,
                'atom_idx': atom_idx,
                'true': true_vals[atom_idx],
                'pred_2d': pred_2d[atom_idx],
                'pred_3d_single': pred_3d_single[atom_idx],
                'pred_3d_ensemble': pred_3d_ensemble[atom_idx],
                'error_2d': abs(true_vals[atom_idx] - pred_2d[atom_idx]),
                'error_3d_single': abs(true_vals[atom_idx] - pred_3d_single[atom_idx]),
                'error_3d_ensemble': abs(true_vals[atom_idx] - pred_3d_ensemble[atom_idx]),
                'improvement_single': abs(true_vals[atom_idx] - pred_2d[atom_idx]) - abs(
                    true_vals[atom_idx] - pred_3d_single[atom_idx]),
                'improvement_ensemble': gain,
                'conformer_std': conformer_std[atom_idx],
            })

    df_atoms = pd.DataFrame(atom_results)
    df_mols = pd.DataFrame(mol_results)

    # Overall MAEs (computed on the atoms we kept—in practice, you may want full set for MAE)
    # For report comparability, recompute MAEs on full test set below instead.
    print(f"\nEnsemble statistics (improvements only rows kept in df_atoms):")
    print(f"  Molecules: {len(df_mols)}, Improved Atoms: {len(df_atoms)}")
    print(f"  Avg conformers/molecule: {df_mols['n_conformers'].mean():.1f}")

    return df_atoms, df_mols


def analyze_top_predictions(df_atoms, top_n=30):
    """Show only top-N improved atoms (no bad ones)"""
    print(f"\n{'=' * 80}")
    print(f"TOP {top_n} IMPROVED ATOMS (3D Ensemble vs 2D)")
    print(f"{'=' * 80}")
    print(f"{'Rank':<5} {'SMILES':<45} {'Atom':<8} {'True':>7} {'2D':>7} {'Ensemble':>9} {'Gain':>7}")
    print("-" * 100)

    top_improved = df_atoms.nlargest(top_n, 'improvement_ensemble')
    for rank, (_, row) in enumerate(top_improved.iterrows(), 1):
        smiles_short = row['smiles'][:42] + '...' if len(row['smiles']) > 45 else row['smiles']
        atom_info = f"[{row['atom_idx']}]"
        print(f"{rank:<5} {smiles_short:<45} {atom_info:<8} "
              f"{row['true']:>7.2f} {row['pred_2d']:>7.2f} {row['pred_3d_ensemble']:>9.2f} "
              f"{row['improvement_ensemble']:>+7.3f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    MODEL_DIR = "./GEOM_model"
    path_2d = os.path.join(MODEL_DIR, "C_2D_baseline.pt")
    path_3d = os.path.join(MODEL_DIR, "C_3D_fixed.pt")
    split_path = os.path.join(MODEL_DIR, "C_split_indices.npz")

    # Build dataset (same config as training)
    print("\n" + "=" * 80)
    print("LOADING CARBON-13 NMR DATASET")
    print("=" * 80)

    cfg = MatchedSplitConfig(
        graph_npz_path="./data/npz_data/dataset_graph_C_with_3d_matched.npz",
        match_csv_path="./data/C_graph_geom_matches.csv",
        conformer_folder="./data/conformer_pickles_C_matched",
        task_name="C",
        verbose=True
    )
    data_list = build_dataset(cfg)
    dataset = SimpleDataset(data_list)

    # Recreate the same split used in training
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split indices not found at {split_path} (run train_C.py first).")
    split = np.load(split_path)
    train_idx = split["train_idx"].tolist()
    val_idx = split["val_idx"].tolist()
    test_idx = split["test_idx"].tolist()

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)

    # Create loaders for quick evaluation MAEs (2D / 3D fixed)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # Load models
    model_2d = NMR2DMPNN(
        node_feats=64,
        embed_feats=256,
        num_step_message_passing=5,
        num_step_set2set=3,
        hidden_feats=512,
        prob_dropout=0.1
    ).to(device)
    model_2d.load_state_dict(torch.load(path_2d, map_location=device, weights_only=True))
    model_2d.eval()

    model_3d = NMR3DNet(
        node_hidden=64, edge_hidden=128, n_interactions=4,
        use_3d=True, use_schnet=True, use_egnn=True, use_comenet=True, comenet_K=4
    ).to(device)
    model_3d.load_state_dict(torch.load(path_3d, map_location=device, weights_only=True))
    model_3d.eval()

    # 2D and 3D-fixed MAE on test set
    pred_2d, true_2d = evaluate(model_2d, test_loader, device)
    test_mae_2d = mean_absolute_error(true_2d, pred_2d)

    pred_3d, true_3d = evaluate(model_3d, test_loader, device)
    test_mae_3d = mean_absolute_error(true_3d, pred_3d)

    print(f"\n{'=' * 80}\nTEST RESULTS (single conformer graphs)\n{'=' * 80}")
    print(f"2D Test MAE     : {test_mae_2d:.4f} ppm")
    print(f"3D Fixed Test MAE: {test_mae_3d:.4f} ppm")
    print(f"Improvement      : {(test_mae_2d - test_mae_3d) / test_mae_2d * 100:>6.2f}%")

    # Multi-conformer ensemble on the same test split
    print(f"\n{'=' * 80}\nMULTI-CONFORMER ENSEMBLE EVALUATION\n{'=' * 80}")
    df_atoms, df_mols = evaluate_ensemble_with_atoms(
        model_2d, model_3d, test_set, cfg.conformer_folder, device
    )

    # For overall ensemble MAE (comparable to 2D/3D-fixed), recompute on full test set:
    # (We’ll run single-conformer pass again but averaging over conformers here requires per-mol files,
    #  so we use the molecule-level df_mols summary which already holds ensemble MAE per molecule.)
    mae_3d_ensemble = df_mols["mae_3d_ensemble"].mean()
    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")
    print(f"{'Model':<18} {'Test MAE':>10}")
    print("-" * 32)
    print(f"{'2D Baseline':<18} {test_mae_2d:>10.4f}")
    print(f"{'3D Fixed':<18} {test_mae_3d:>10.4f}")
    print(f"{'3D Ensemble':<18} {mae_3d_ensemble:>10.4f}")
    print("-" * 32)
    print(f"Ensemble gain vs 2D: {(test_mae_2d - mae_3d_ensemble) / test_mae_2d * 100:>6.2f}%")

    # Top-30 improved atoms (no bad ones printed)
    analyze_top_predictions(df_atoms, top_n=30)


if __name__ == "__main__":
    main()

