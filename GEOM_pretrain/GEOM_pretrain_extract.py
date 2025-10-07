# find_similar_for_pretraining_fast.py (FIXED)
"""
Fast similarity search - pre-compute all fingerprints
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def canonicalize_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def get_heavy_atom_count(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumHeavyAtoms() if mol else None
    except:
        return None


def compute_fingerprint(smiles, fp_size=2048, radius=2):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
        return gen.GetFingerprint(mol)
    except:
        return None


def load_geom_summary(json_path, source_name):
    print(f"Loading {source_name}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    geom_smiles = []
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = [(d.get('smiles', ''), d) for d in data]
    else:
        return []

    for smiles, info in items:
        if not smiles and isinstance(info, dict):
            smiles = info.get('smiles') or info.get('SMILES')
        if smiles:
            canonical = canonicalize_smiles(smiles)
            if canonical:
                geom_smiles.append({
                    'smiles': canonical,
                    'original_smiles': smiles,
                    'source': source_name
                })

    print(f"  Loaded: {len(geom_smiles):,}")
    return geom_smiles


def main():
    # ========== PATHS ==========
    NPZ_PATH = r"E:\GEOM\data_competition_1d_code_data\data\dataset_graph_C.npz"
    MATCHED_CSV = r"E:\GEOM\data_competition_1d_code_data\data\C_graph_geom_matches.csv"
    JSON_QM9 = r"E:\rdkit_folder\rdkit_folder\summary_qm9.json"
    JSON_DRUGS = r"E:\rdkit_folder\rdkit_folder\summary_drugs.json"
    OUTPUT_CSV = "./C_pretraining_similar_molecules.csv"
    # ===========================

    # ========== PARAMETERS ==========
    SIMILARITY_THRESHOLD = 0.6
    TOP_K = 3
    ATOM_TOLERANCE = 2
    MAX_QUERIES = None  # Start with 5K for testing (30 min), then set to None for all
    # ================================

    print(f"\n{'=' * 80}")
    print("FAST SIMILARITY SEARCH")
    print(f"{'=' * 80}")

    # Step 1: Load unmatched
    print("\n[1/6] Loading unmatched NMR...")
    pack = np.load(NPZ_PATH, allow_pickle=True)
    mol_dict = pack['data'].item()
    all_smiles = [str(s) for s in mol_dict['smi']]

    df_matched = pd.read_csv(MATCHED_CSV)
    matched_indices = set(df_matched['graph_idx'].tolist())

    unmatched = []
    for idx, smi in enumerate(all_smiles):
        if idx not in matched_indices:
            canonical = canonicalize_smiles(smi)
            if canonical:
                unmatched.append({'graph_idx': idx, 'smiles': canonical})

    if MAX_QUERIES:
        import random
        random.seed(42)
        unmatched = random.sample(unmatched, min(MAX_QUERIES, len(unmatched)))

    print(f"  Processing {len(unmatched):,} queries")

    # Step 2: Load GEOM
    print("\n[2/6] Loading GEOM...")
    geom_molecules = []
    if Path(JSON_QM9).exists():
        geom_molecules.extend(load_geom_summary(JSON_QM9, 'qm9'))
    if Path(JSON_DRUGS).exists():
        geom_molecules.extend(load_geom_summary(JSON_DRUGS, 'drugs'))
    print(f"  Total: {len(geom_molecules):,}")

    # Step 3: Pre-compute atom counts
    print("\n[3/6] Computing atom counts...")
    query_atoms = [get_heavy_atom_count(d['smiles']) for d in tqdm(unmatched, desc="  Queries")]
    target_atoms = [get_heavy_atom_count(d['smiles']) for d in tqdm(geom_molecules, desc="  Targets")]

    # Build atom index
    atom_index = defaultdict(list)
    for idx, n_atoms in enumerate(target_atoms):
        if n_atoms:
            atom_index[n_atoms].append(idx)

    # Step 4: Pre-compute ALL fingerprints once
    print("\n[4/6] Pre-computing ALL fingerprints...")
    print("  Query fingerprints...")
    query_fps = [compute_fingerprint(d['smiles']) for d in tqdm(unmatched)]

    print("  Target fingerprints...")
    target_fps = [compute_fingerprint(d['smiles']) for d in tqdm(geom_molecules)]

    print(f"  Query FPs: {sum(1 for f in query_fps if f):,}/{len(query_fps):,}")
    print(f"  Target FPs: {sum(1 for f in target_fps if f):,}/{len(target_fps):,}")

    # Step 5: Fast similarity search
    print("\n[5/6] Computing similarities...")

    results = []
    stats = {'filtered': 0, 'compared': 0}

    for q_idx in tqdm(range(len(unmatched))):
        if query_fps[q_idx] is None or query_atoms[q_idx] is None:
            continue

        # Get candidates with similar atom count
        candidates = []
        for n in range(query_atoms[q_idx] - ATOM_TOLERANCE,
                       query_atoms[q_idx] + ATOM_TOLERANCE + 1):
            candidates.extend(atom_index.get(n, []))

        stats['filtered'] += len(candidates)

        # Compare with candidates only
        similarities = []
        for t_idx in candidates:
            if target_fps[t_idx] is None:
                continue

            sim = DataStructs.TanimotoSimilarity(query_fps[q_idx], target_fps[t_idx])
            stats['compared'] += 1

            if sim >= SIMILARITY_THRESHOLD:
                similarities.append((t_idx, sim))

        if similarities:
            similarities.sort(key=lambda x: x[1], reverse=True)
            for t_idx, sim in similarities[:TOP_K]:
                results.append({
                    'nmr_graph_idx': unmatched[q_idx]['graph_idx'],
                    'nmr_smiles': unmatched[q_idx]['smiles'],
                    'geom_smiles': geom_molecules[t_idx]['smiles'],
                    'geom_original_smiles': geom_molecules[t_idx]['original_smiles'],
                    'similarity': sim,
                    'source': geom_molecules[t_idx]['source']
                })

    df_results = pd.DataFrame(results)
    unique_geom = df_results.drop_duplicates('geom_smiles') if len(df_results) > 0 else pd.DataFrame()

    # Results
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    print(f"Possible comparisons:  {len(unmatched) * len(geom_molecules):,}")
    print(f"After filtering:       {stats['filtered']:,}")
    print(f"Actual comparisons:    {stats['compared']:,}")
    print(f"Speedup:               {(len(unmatched) * len(geom_molecules)) / max(stats['compared'], 1):.0f}x")
    print(f"\nMatches:               {len(df_results):,}")
    print(f"Unique GEOM:           {len(unique_geom):,}")

    if len(df_results) > 0:
        print(f"\nSimilarity: {df_results['similarity'].mean():.3f} ± {df_results['similarity'].std():.3f}")
        print(f"\nSources:\n{unique_geom['source'].value_counts()}")

        df_results.to_csv(OUTPUT_CSV, index=False)
        unique_geom.to_csv("./C_unique_geom_for_pretraining.csv", index=False)
        print(f"\n✓ Saved results")


if __name__ == "__main__":
    main()