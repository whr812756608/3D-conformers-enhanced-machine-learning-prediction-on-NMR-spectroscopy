# train_C_with_ensemble_evaluation.py

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
from Model_3D_NMR_hybrid import NMR2DMPNN, NMR3DNet, evaluate

torch.manual_seed(0)
np.random.seed(0)


class SimpleDataset(TorchDataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def train_epoch(model, loader, optimizer, device, y_mean, y_std, lambda_reg=0.02):
    """Training with optional regularization for adaptive models"""
    model.train()
    total_loss = 0
    total_pred_loss = 0
    total_reg_loss = 0

    for data in loader:
        data = data.to(device)
        pred = model(data)
        target = data.y[data.mask]

        pred_norm = (pred - y_mean) / y_std
        target_norm = (target - y_mean) / y_std

        pred_loss = torch.nn.functional.l1_loss(pred_norm, target_norm)
        loss = pred_loss

        # Regularization for adaptive models
        if hasattr(model, 'selector') and hasattr(model, 'use_adaptive') and model.use_adaptive:
            weights = model.selector(data.x)
            mean_weight = weights.mean()
            reg_loss = lambda_reg * (0.65 - mean_weight).clamp(min=0) ** 2
            loss = loss + reg_loss
            total_reg_loss += reg_loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        total_pred_loss += pred_loss.item()

    n = len(loader)
    return total_loss / n, total_pred_loss / n, total_reg_loss / n if total_reg_loss > 0 else 0



@torch.no_grad()
def diagnose_adaptive_model(model, loader, device):
    """Diagnose what the adaptive selector is learning"""
    if not hasattr(model, 'get_mixing_weights'):
        return

    model.eval()
    all_weights = []
    atom_types = []

    from rdkit import Chem

    for batch in loader:
        batch = batch.to(device)

        # Get all weights for this batch
        weights = model.selector(batch.x).cpu().numpy().flatten()

        # Handle batched SMILES (it's a list)
        smiles_list = batch.smiles if isinstance(batch.smiles, list) else [batch.smiles]

        # Track which atoms belong to which molecule
        batch_indices = batch.batch.cpu().numpy()

        # Process each molecule in the batch
        current_atom_idx = 0
        for mol_idx, smiles in enumerate(smiles_list):
            # Get weights for this molecule's atoms
            mol_mask = (batch_indices == mol_idx)
            mol_weights = weights[mol_mask]
            all_weights.extend(mol_weights.tolist())

            # Get atom types
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
                for i, atom in enumerate(atoms):
                    if i < len(mol_weights):
                        atom_types.append({
                            'symbol': atom.GetSymbol(),
                            'hybrid': str(atom.GetHybridization()),
                            'weight': mol_weights[i]
                        })

    if not all_weights:
        return

    all_weights = np.array(all_weights)

    print(f"\n{'=' * 80}")
    print("ADAPTIVE SELECTOR DIAGNOSIS")
    print(f"{'=' * 80}")
    print(f"Total atoms: {len(all_weights)}")
    print(f"\n3D Usage Statistics:")
    print(f"  Mean: {all_weights.mean():.4f}")
    print(f"  Std:  {all_weights.std():.4f}")
    print(f"  Min:  {all_weights.min():.4f}")
    print(f"  Max:  {all_weights.max():.4f}")

    print(f"\nUsage Distribution:")
    for label, low, high in [("Strongly 2D", 0.0, 0.2), ("Mostly 2D", 0.2, 0.4),
                             ("Mixed", 0.4, 0.6), ("Mostly 3D", 0.6, 0.8),
                             ("Strongly 3D", 0.8, 1.0)]:
        count = ((all_weights >= low) & (all_weights <= high)).sum()
        pct = count / len(all_weights) * 100
        print(f"  {label:<15} ({low:.1f}-{high:.1f}): {count:>6} ({pct:>5.1f}%)")

    if atom_types:
        df = pd.DataFrame(atom_types)
        print(f"\n3D Usage by Atom Type:")
        atom_stats = df.groupby('symbol')['weight'].agg(['count', 'mean', 'std']).sort_values('mean')
        print(f"{'Atom':<6} {'Count':>8} {'Mean 3D':>10} {'Std':>8}")
        print("-" * 40)
        for atom, row in atom_stats.iterrows():
            print(f"{atom:<6} {int(row['count']):>8} {row['mean']:>10.4f} {row['std']:>8.4f}")

    print(f"{'=' * 80}\n")


def load_pretrained_weights(model, pretrained_path, device, strict=False):
    """Load pretrained 3D encoder weights"""
    if not os.path.exists(pretrained_path):
        print(f"⚠ Pretrained weights not found: {pretrained_path}")
        print("  Training from scratch...")
        return model

    print(f"\n{'=' * 80}")
    print(f"LOADING PRETRAINED WEIGHTS")
    print(f"{'=' * 80}")
    print(f"Path: {pretrained_path}")

    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)

    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print(f"✓ Successfully loaded pretrained weights")
        print(f"  Pretrained epoch: {checkpoint['epoch']}")
        print(f"  Pretrained loss: {checkpoint['loss']:.4f}")

        if 'stats' in checkpoint:
            print(f"  Pretrained on {checkpoint['stats']['success']} conformers")

    except RuntimeError as e:
        if strict:
            raise e
        else:
            print(f"⚠ Loading with strict=False")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    print(f"{'=' * 80}\n")
    return model


def train_model(config_name, model, dataset, device, train_set, val_set, test_set,
                epochs=100, is_adaptive=False, lambda_reg=0.02):
    """Train model with fixed splits"""
    print(f"\n{'=' * 80}\nTraining: {config_name}\n{'=' * 80}")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    print(f"Split sizes: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    # Normalization stats
    train_y_list = []
    for batch in train_loader:
        train_y_list.append(batch.y[batch.mask].cpu().numpy())
    train_y = np.hstack(train_y_list)
    y_mean = torch.tensor(train_y.mean(), dtype=torch.float32, device=device)
    y_std = torch.tensor(train_y.std(), dtype=torch.float32, device=device)

    print(f"Training labels: mean={y_mean:.2f}, std={y_std:.2f}, n={len(train_y)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Use lower LR for adaptive models
    if is_adaptive:
        lr = 5e-4
        print(f"Adaptive model: using lr={lr}, lambda_reg={lambda_reg}")
    else:
        lr = 1e-3

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 10, min_lr=1e-6)

    best_val_mae = float('inf')
    patience = 0

    for epoch in range(1, epochs + 1):
        train_loss, pred_loss, reg_loss = train_epoch(
            model, train_loader, optimizer, device, y_mean, y_std, lambda_reg
        )
        val_pred, val_true = evaluate(model, val_loader, device)
        val_mae = mean_absolute_error(val_true, val_pred)
        scheduler.step(val_mae)

        if epoch % 20 == 0:
            if is_adaptive and reg_loss > 0:
                print(
                    f"  Epoch {epoch:3d}: loss={train_loss:.4f} (pred={pred_loss:.4f}, reg={reg_loss:.5f}), val={val_mae:.4f}")
            else:
                print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}, val={val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), f"./GEOM_model/{config_name}.pt")
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                print(f"  Early stop at epoch {epoch}")
                break

    # Test
    model.load_state_dict(torch.load(f"./GEOM_model/{config_name}.pt", weights_only=True))
    test_pred, test_true = evaluate(model, test_loader, device)
    test_mae = mean_absolute_error(test_true, test_pred)

    print(f"Results: val={best_val_mae:.4f}, test={test_mae:.4f} ppm")

    # Diagnose adaptive model
    if is_adaptive:
        diagnose_adaptive_model(model, test_loader, device)

    return model, best_val_mae, test_mae


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
        except:
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

        # Prepare data
        data_dev = data.to(device)
        data_dev.batch = torch.zeros(data_dev.x.size(0), dtype=torch.long, device=device)

        # 2D prediction
        pred_2d = model_2d(data_dev).cpu().numpy()

        # 3D predictions for ALL conformers
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

        # Ensemble
        conformer_preds = np.array(conformer_preds)
        conformer_weights = np.array(conformer_weights).reshape(-1, 1)
        conformer_weights = conformer_weights / conformer_weights.sum()

        pred_3d_ensemble = (conformer_preds * conformer_weights).sum(axis=0)
        pred_3d_single = conformer_preds[0]

        # Ground truth
        true_vals = data.y[data.mask].cpu().numpy()
        n_masked = len(true_vals)

        # Get atom types
        mol = Chem.MolFromSmiles(smiles)
        atom_symbols = []
        atom_hybrids = []
        if mol:
            atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
            for i in range(min(n_masked, len(atoms))):
                atom_symbols.append(atoms[i].GetSymbol())
                atom_hybrids.append(str(atoms[i].GetHybridization()))
        else:
            atom_symbols = ['?'] * n_masked
            atom_hybrids = ['?'] * n_masked

        # Molecule-level
        mol_results.append({
            'smiles': smiles,
            'n_atoms': n_masked,
            'n_conformers': len(conformer_preds),
            'mae_2d': np.abs(true_vals - pred_2d).mean(),
            'mae_3d_single': np.abs(true_vals - pred_3d_single).mean(),
            'mae_3d_ensemble': np.abs(true_vals - pred_3d_ensemble).mean()
        })

        # Atom-level
        conformer_std = conformer_preds.std(axis=0) if len(conformer_preds) > 1 else np.zeros(n_masked)

        for atom_idx in range(n_masked):
            atom_results.append({
                'smiles': smiles,
                'atom_idx': atom_idx,
                'atom_symbol': atom_symbols[atom_idx],
                'atom_hybrid': atom_hybrids[atom_idx],
                'n_conformers': len(conformer_preds),
                'conformer_std': conformer_std[atom_idx],
                'true': true_vals[atom_idx],
                'pred_2d': pred_2d[atom_idx],
                'pred_3d_single': pred_3d_single[atom_idx],
                'pred_3d_ensemble': pred_3d_ensemble[atom_idx],
                'error_2d': abs(true_vals[atom_idx] - pred_2d[atom_idx]),
                'error_3d_single': abs(true_vals[atom_idx] - pred_3d_single[atom_idx]),
                'error_3d_ensemble': abs(true_vals[atom_idx] - pred_3d_ensemble[atom_idx]),
                'improvement_single': abs(true_vals[atom_idx] - pred_2d[atom_idx]) - abs(
                    true_vals[atom_idx] - pred_3d_single[atom_idx]),
                'improvement_ensemble': abs(true_vals[atom_idx] - pred_2d[atom_idx]) - abs(
                    true_vals[atom_idx] - pred_3d_ensemble[atom_idx])
            })

    df_atoms = pd.DataFrame(atom_results)
    df_mols = pd.DataFrame(mol_results)

    mae_2d = df_atoms['error_2d'].mean()
    mae_3d_single = df_atoms['error_3d_single'].mean()
    mae_3d_ensemble = df_atoms['error_3d_ensemble'].mean()

    print(f"\nEnsemble statistics:")
    print(f"  Molecules: {len(df_mols)}, Atoms: {len(df_atoms)}")
    print(f"  Avg conformers/molecule: {df_mols['n_conformers'].mean():.1f}")

    return mae_2d, mae_3d_single, mae_3d_ensemble, df_atoms, df_mols


def analyze_top_predictions(df_atoms, top_n=30):
    """Analyze top improved atoms"""
    print(f"\n{'=' * 80}")
    print(f"TOP {top_n} IMPROVED ATOMS (3D Ensemble vs 2D)")
    print(f"{'=' * 80}")
    print(f"{'Rank':<5} {'SMILES':<45} {'Atom':<8} {'Type':<6} {'True':>7} {'2D':>7} {'Ensemble':>9} {'Gain':>7}")
    print("-" * 120)

    top_improved = df_atoms.nlargest(top_n, 'improvement_ensemble')
    for rank, (_, row) in enumerate(top_improved.iterrows(), 1):
        smiles_short = row['smiles'][:42] + '...' if len(row['smiles']) > 45 else row['smiles']
        atom_info = f"{row['atom_symbol']}[{row['atom_idx']}]"
        print(f"{rank:<5} {smiles_short:<45} {atom_info:<8} {row['atom_hybrid']:<6} "
              f"{row['true']:>7.2f} {row['pred_2d']:>7.2f} {row['pred_3d_ensemble']:>9.2f} "
              f"{row['improvement_ensemble']:>+7.3f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("./GEOM_model", exist_ok=True)

    # Pretrained weights
    PRETRAINED_PATH = "./GEOM_pretrain/pretrained_models/C_pretrained.pt"
    USE_PRETRAINED = False

    # Build dataset
    print("\n" + "=" * 80)
    print("LOADING CARBON-13 NMR DATASET")
    print("=" * 80)

    cfg = MatchedSplitConfig(
        graph_npz_path="./data/data/dataset_graph_C_with_3d_matched.npz",
        match_csv_path="./data/C_graph_geom_matches.csv",
        conformer_folder="./data/conformer_pickles_C_matched",
        task_name="C",
        verbose=True
    )
    data_list = build_dataset(cfg)
    dataset = SimpleDataset(data_list)

    # Fixed split
    n = len(dataset)
    torch.manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [int(0.8 * n), int(0.1 * n), n - int(0.8 * n) - int(0.1 * n)]
    )

    print(f"\nDataset split: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test")

    # ========== Train 2D baseline ==========
    print(f"\n{'=' * 80}")
    print("TRAINING 2D BASELINE")
    print(f"{'=' * 80}")

    # model_2d = NMR3DNet(
    #     node_hidden=64, edge_hidden=128, n_interactions=4,
    #     use_3d=False, use_schnet=True, use_egnn=False
    # ).to(device)

    model_2d = NMR2DMPNN(
        node_feats=64,
        embed_feats=256,
        num_step_message_passing=5,
        num_step_set2set=3,
        hidden_feats=512,
        prob_dropout=0.1
    ).to(device)

    model_2d, val_2d, test_2d = train_model(
        "C_2D_baseline", model_2d, dataset, device, train_set, val_set, test_set,
        is_adaptive=False
    )

    # ========== Train 3D fixed ==========
    print(f"\n{'=' * 80}")
    print("TRAINING 3D FIXED MODEL")
    print(f"{'=' * 80}")

    model_3d = NMR3DNet(
        node_hidden=64, edge_hidden=128, n_interactions=4,
        use_3d=True, use_schnet=True, use_egnn=True, use_comenet= True
    ).to(device)

    if USE_PRETRAINED:
        model_3d = load_pretrained_weights(model_3d, PRETRAINED_PATH, device, strict=False)

    model_3d, val_3d, test_3d = train_model(
        "C_3D_fixed", model_3d, dataset, device, train_set, val_set, test_set,
        is_adaptive=False
    )

    # Ensemble evaluation
    print(f"\n{'=' * 80}\nMULTI-CONFORMER ENSEMBLE EVALUATION\n{'=' * 80}")
    mae_2d, mae_3d_single, mae_3d_ensemble, df_atoms, df_mols = evaluate_ensemble_with_atoms(
        model_2d, model_3d, test_set, cfg.conformer_folder, device
    )

    # Save results
    df_atoms.to_csv("C_atom_level_analysis.csv", index=False)
    df_mols.to_csv("C_molecule_level_analysis.csv", index=False)
    print("\nSaved: C_atom_level_analysis.csv, C_molecule_level_analysis.csv")

    # Summary
    print(f"\n{'=' * 80}\nOVERALL RESULTS\n{'=' * 80}")
    print(f"{'Model':<25} {'Val MAE':>10} {'Test MAE':>10} {'Improvement':>12}")
    print("-" * 80)
    print(f"{'2D Baseline':<25} {val_2d:>10.4f} {test_2d:>10.4f} {'baseline':>12}")
    print(f"{'3D Fixed':<25} {val_3d:>10.4f} {test_3d:>10.4f} {(test_2d - test_3d) / test_2d * 100:>11.1f}%")
    # print(
    #     f"{'3D Adaptive':<25} {val_adaptive:>10.4f} {test_adaptive:>10.4f} {(test_2d - test_adaptive) / test_2d * 100:>11.1f}%")
    print(
        f"{'3D Ensemble':<25} {'-':>10} {mae_3d_ensemble:>10.4f} {(test_2d - mae_3d_ensemble) / test_2d * 100:>11.1f}%")
    print(f"{'=' * 80}")

    # Quick statistics
    print(f"\nQuick Statistics:")
    print(
        f"  Atoms improved: {(df_atoms['improvement_ensemble'] > 0).sum()} / {len(df_atoms)} ({(df_atoms['improvement_ensemble'] > 0).sum() / len(df_atoms) * 100:.1f}%)")
    print(f"  Mean improvement: {df_atoms['improvement_ensemble'].mean():.4f} ppm")
    print(f"  Median improvement: {df_atoms['improvement_ensemble'].median():.4f} ppm")

    # Detailed analysis
    analyze_top_predictions(df_atoms, top_n=30)


if __name__ == "__main__":
    main()