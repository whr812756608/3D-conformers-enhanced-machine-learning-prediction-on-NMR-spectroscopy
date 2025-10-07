# graph_builder_with_matched_split.py (CORRECTED - SIMPLE APPROACH)

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import torch
from torch_geometric.data import Data
from rdkit import Chem


def canonical_atom_order(mol: Chem.Mol) -> List[int]:
    """Get canonical atom ordering (heavy atoms only)"""
    mol_no_h = Chem.RemoveHs(mol, sanitize=False)
    ranks = list(Chem.CanonicalRankAtoms(mol_no_h, includeChirality=True))
    return sorted(range(len(ranks)), key=lambda i: ranks[i])


def sanitize_filename(smiles):
    """Sanitize SMILES for filename"""
    return smiles.replace('/', '_').replace('\\', '_').replace(':', '_')


def canonicalize_remove_h(smiles):
    """Canonicalize SMILES after removing explicit [H]"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.RemoveHs(mol, sanitize=False)
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def match_conformer_to_graph(graph_smiles: str, conf_mol: Chem.Mol, expected_count: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Match conformer atoms to graph atom ordering"""
    if conf_mol is None or conf_mol.GetNumConformers() == 0:
        return None

    try:
        graph_mol = Chem.MolFromSmiles(graph_smiles)
        if graph_mol is None:
            return None

        conf_mol_no_h = Chem.RemoveHs(conf_mol, sanitize=False)
        graph_mol_no_h = Chem.RemoveHs(graph_mol, sanitize=False)

        if conf_mol_no_h.GetNumAtoms() != expected_count:
            return None
        if graph_mol_no_h.GetNumAtoms() != expected_count:
            return None

        conf_canonical_order = canonical_atom_order(conf_mol)
        graph_canonical_order = canonical_atom_order(graph_mol)

        inverse_conf = {canon_idx: conf_idx for conf_idx, canon_idx in enumerate(conf_canonical_order)}
        permutation = np.array([inverse_conf[canon_idx] for canon_idx in graph_canonical_order])

        non_h_indices = [i for i, atom in enumerate(conf_mol.GetAtoms())
                         if atom.GetAtomicNum() != 1]
        all_coords = conf_mol.GetConformer(0).GetPositions()
        conf_coords = all_coords[non_h_indices]
        graph_coords = conf_coords[permutation]

        return graph_coords.astype(np.float32), permutation
    except:
        return None


@dataclass
class MatchedSplitConfig:
    """Configuration"""
    graph_npz_path: str
    match_csv_path: str
    conformer_folder: str
    task_name: str
    verbose: bool = True


def build_dataset(cfg: MatchedSplitConfig):
    """Build dataset - NPZ already contains only heavy atoms"""

    if cfg.verbose:
        print(f"\n{'=' * 80}")
        print(f"BUILDING {cfg.task_name} NMR DATASET")
        print(f"{'=' * 80}")

    # Load NPZ
    pack = np.load(cfg.graph_npz_path, allow_pickle=True)
    mol_dict = pack['data'].item()
    n_node = np.asarray(mol_dict['n_node'])
    n_molecules = len(n_node)

    # Load matches
    df_matches = pd.read_csv(cfg.match_csv_path)

    if cfg.verbose:
        print(f"  NPZ molecules: {n_molecules}")
        print(f"  Match entries: {len(df_matches)}")

    # Build pickle lookup
    conformer_folder = Path(cfg.conformer_folder)
    pickle_lookup = {f.stem: str(f) for f in conformer_folder.glob("*.pickle")}
    print(f"  Pickle files: {len(pickle_lookup)}")

    # Map indices
    original_indices = sorted(df_matches['graph_idx'].tolist())
    orig_to_split = {orig: split for split, orig in enumerate(original_indices)}

    # Build conformer map
    conformer_map = {}
    for _, row in df_matches.iterrows():
        orig_idx = row['graph_idx']
        if orig_idx not in orig_to_split:
            continue

        split_idx = orig_to_split[orig_idx]
        graph_smiles = row['graph_smiles']

        safe_smiles = sanitize_filename(graph_smiles)
        canonical_no_h = canonicalize_remove_h(graph_smiles)
        safe_canonical = sanitize_filename(canonical_no_h) if canonical_no_h else None

        for candidate in [
            f"{safe_smiles}_{orig_idx}",
            safe_smiles,
            f"{safe_canonical}_{orig_idx}" if safe_canonical else None,
            safe_canonical
        ]:
            if candidate and candidate in pickle_lookup:
                conformer_map[split_idx] = pickle_lookup[candidate]
                break

    print(f"  Mapped: {len(conformer_map)}/{n_molecules}")

    # Statistics
    stats = {
        'attempted': 0,
        'success': 0,
        'no_pickle': 0,
        'load_error': 0,
        'no_conformers': 0,
        'atom_mismatch': 0
    }

    conformer_stats = []
    failures = []

    n_ptr = np.concatenate([[0], np.cumsum(n_node)])
    e_ptr = np.concatenate([[0], np.cumsum(mol_dict['n_edge'])])
    data_list = []

    print("\nProcessing...")
    for split_idx in range(n_molecules):
        stats['attempted'] += 1

        if cfg.verbose and (split_idx + 1) % 500 == 0:
            print(f"  {split_idx + 1}/{n_molecules}... (success: {stats['success']})")

        n0, n1 = int(n_ptr[split_idx]), int(n_ptr[split_idx + 1])
        e0, e1 = int(e_ptr[split_idx]), int(e_ptr[split_idx + 1])
        N = int(n_node[split_idx])
        smiles = str(mol_dict['smi'][split_idx])

        # Get pickle
        pickle_path = conformer_map.get(split_idx)
        if pickle_path is None:
            stats['no_pickle'] += 1
            failures.append({'split_idx': split_idx, 'smiles': smiles, 'failure_type': 'no_pickle'})
            continue

        # Load conformer
        try:
            with open(pickle_path, 'rb') as f:
                conf_data = pickle.load(f)
        except Exception as e:
            stats['load_error'] += 1
            failures.append({'split_idx': split_idx, 'smiles': smiles, 'failure_type': 'load_error', 'error': str(e)})
            continue

        conformers = conf_data.get('conformers', [])
        if not conformers:
            stats['no_conformers'] += 1
            failures.append({'split_idx': split_idx, 'smiles': smiles, 'failure_type': 'no_conformers'})
            continue

        # Best conformer (highest Boltzmann weight)
        weights = [c.get('boltzmannweight', 0) for c in conformers]
        best_idx = np.argmax(weights)
        best_conf = conformers[best_idx]
        rd_mol = best_conf.get('rd_mol')

        # Match conformer to graph
        match_result = match_conformer_to_graph(smiles, rd_mol, N)
        if match_result is None:
            stats['atom_mismatch'] += 1
            failures.append({'split_idx': split_idx, 'smiles': smiles, 'failure_type': 'atom_mismatch'})
            continue

        pos_np, _ = match_result

        # SIMPLE: NPZ already contains only heavy atoms, use direct slicing
        x = torch.as_tensor(mol_dict['node_attr'][n0:n1].astype(np.float32))
        y = torch.as_tensor(mol_dict['shift'][n0:n1], dtype=torch.float32)
        mask = torch.as_tensor(mol_dict['mask'][n0:n1], dtype=torch.bool)

        src_local = torch.as_tensor(mol_dict['src'][e0:e1], dtype=torch.long) - n0
        dst_local = torch.as_tensor(mol_dict['dst'][e0:e1], dtype=torch.long) - n0
        valid = (src_local >= 0) & (src_local < N) & (dst_local >= 0) & (dst_local < N)
        edge_index = torch.stack([src_local[valid], dst_local[valid]], dim=0)
        edge_attr = torch.as_tensor(mol_dict['edge_attr'][e0:e1][valid.numpy()].astype(np.float32))

        pos = torch.as_tensor(pos_np, dtype=torch.float32)

        # Verify dimensions
        assert x.size(0) == N, f"x mismatch: {x.size(0)} != {N}"
        assert pos.size(0) == N, f"pos mismatch: {pos.size(0)} != {N}"

        conformer_stats.append({
            'split_idx': split_idx,
            'smiles': smiles,
            'n_conformers': len(conformers),
            'selected_boltzmann_weight': weights[best_idx],
            'n_atoms': N
        })

        data_list.append(Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            y=y, mask=mask, smiles=smiles, pos=pos
        ))
        stats['success'] += 1

    # Save reports
    df_failures = pd.DataFrame(failures)
    df_failures.to_csv(f"{cfg.task_name}_NMR_failure_report.csv", index=False)

    df_conformer_stats = pd.DataFrame(conformer_stats)
    df_conformer_stats.to_csv(f"{cfg.task_name}_NMR_conformer_stats.csv", index=False)

    # Print statistics
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    print(f"Success: {stats['success']}/{n_molecules} ({stats['success'] / n_molecules * 100:.1f}%)")
    print(f"\nFailures:")
    print(f"  No pickle:      {stats['no_pickle']}")
    print(f"  Load error:     {stats['load_error']}")
    print(f"  No conformers:  {stats['no_conformers']}")
    print(f"  Atom mismatch:  {stats['atom_mismatch']}")

    if len(df_conformer_stats) > 0:
        print(f"\n3D Statistics:")
        print(f"  Total molecules: {len(df_conformer_stats)}")
        print(f"  Avg conformers/mol: {df_conformer_stats['n_conformers'].mean():.1f}")
        print(f"  Total atoms with 3D: {df_conformer_stats['n_atoms'].sum()}")

    print(f"\nReports saved: {cfg.task_name}_NMR_*.csv")

    return data_list


if __name__ == "__main__":
    cfg_c = MatchedSplitConfig(
        graph_npz_path="npz_data/dataset_graph_C_with_3d_matched.npz",
        match_csv_path="./C_graph_geom_matches.csv",
        conformer_folder="./conformer_pickles_C_matched",
        task_name="C",
        verbose=True
    )
    data_c = build_dataset(cfg_c)
    print(f"\nFinal: {len(data_c)} molecules with 3D coordinates")