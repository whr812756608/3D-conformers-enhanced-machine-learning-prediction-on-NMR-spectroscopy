"""
GEOM_3D_pretraining.py - Self-supervised 3D encoder pretraining

Self-supervised pretraining to specialize the 3D part of NMR3DNet.
The 2D backbone is frozen; 3D heads are trained on unlabeled conformers.

This script is called by the pipeline runner and uses centralized config.
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from rdkit import Chem

# Import central configuration
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import GEOM_13C_NMR.C_NMR_config as config
from GEOM_models.Model_2D_3D_hybrid import NMR3DNet, compute_node_angle_features

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ===========================================================
# Data Loading Utilities
# ===========================================================

def canonical_atom_order(mol):
    """Get canonical atom ordering (heavy atoms only)"""
    mol_no_h = Chem.RemoveHs(mol, sanitize=False)
    ranks = list(Chem.CanonicalRankAtoms(mol_no_h, includeChirality=True))
    return sorted(range(len(ranks)), key=lambda i: ranks[i])


class SimpleDataset(TorchDataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def load_conformer_pickles(conformer_folder, node_dim=74, edge_dim=9):
    """Load conformer pickles -> PyG Data list with x, edge_index, edge_attr, pos, mask"""
    print(f"\n{'=' * 80}\nLOADING CONFORMERS FOR PRETRAINING\n{'=' * 80}")
    print(f"Folder: {conformer_folder}")

    conformer_folder = Path(conformer_folder)
    pickle_files = list(conformer_folder.glob("*.pickle"))
    if len(pickle_files) == 0:
        raise ValueError(f"No pickle files found in {conformer_folder}")
    print(f"Found {len(pickle_files)} pickle files")

    data_list = []
    stats = {
        'attempted': 0, 'success': 0, 'no_conformers': 0,
        'no_mol': 0, 'coord_error': 0, 'other_error': 0
    }

    for pfile in tqdm(pickle_files, desc="Loading"):
        stats['attempted'] += 1
        try:
            with open(pfile, 'rb') as f:
                conf_data = pickle.load(f)

            smiles = conf_data.get('smiles')
            conformers = conf_data.get('conformers', [])
            if not conformers:
                stats['no_conformers'] += 1
                continue

            # Best conformer by Boltzmann weight
            weights = [c.get('boltzmannweight', 0) for c in conformers]
            best_conf = conformers[int(np.argmax(weights))]
            rd_mol = best_conf.get('rd_mol')

            if rd_mol is None or rd_mol.GetNumConformers() == 0:
                stats['no_mol'] += 1
                continue

            # Remove hydrogens
            mol = Chem.RemoveHs(rd_mol, sanitize=False)
            n_atoms = mol.GetNumAtoms()
            if n_atoms == 0:
                stats['no_mol'] += 1
                continue

            # Node features
            x = []
            for atom in mol.GetAtoms():
                feat = [0.0] * node_dim
                feat[0] = atom.GetAtomicNum() / 100.0
                feat[1] = atom.GetDegree() / 4.0
                feat[2] = atom.GetFormalCharge()
                feat[3] = float(atom.GetIsAromatic())
                feat[4] = atom.GetTotalNumHs() / 4.0
                x.append(feat)
            x = torch.tensor(x, dtype=torch.float32)

            # Edge features
            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_index.extend([[i, j], [j, i]])
                e = [0.0] * edge_dim
                e[0] = bond.GetBondTypeAsDouble() / 3.0
                e[1] = float(bond.GetIsAromatic())
                e[2] = float(bond.IsInRing())
                edge_attr.extend([e, e])

            if len(edge_index) == 0:
                stats['other_error'] += 1
                continue

            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

            # 3D coordinates with canonical ordering
            non_h_indices = [i for i, a in enumerate(rd_mol.GetAtoms()) if a.GetAtomicNum() != 1]
            all_coords = rd_mol.GetConformer(0).GetPositions()
            pos_coords = all_coords[non_h_indices]
            canonical_order = canonical_atom_order(rd_mol)
            pos_reordered = pos_coords[canonical_order]
            pos = torch.tensor(pos_reordered, dtype=torch.float32)

            if pos.size(0) != n_atoms or x.size(0) != n_atoms:
                stats['coord_error'] += 1
                continue

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                smiles=smiles,
                y=torch.zeros(n_atoms),
                mask=torch.ones(n_atoms, dtype=torch.bool),
            )
            data_list.append(data)
            stats['success'] += 1

        except Exception:
            stats['other_error'] += 1

    print(f"\n{'=' * 80}\nLOADING STATISTICS\n{'=' * 80}")
    print(f"Attempted: {stats['attempted']}")
    print(f"✓ Success: {stats['success']} ({stats['success'] / max(1, stats['attempted']) * 100:.1f}%)")
    print("Failures:")
    print(f"  No conformers: {stats['no_conformers']}")
    print(f"  No molecule:   {stats['no_mol']}")
    print(f"  Coord error:   {stats['coord_error']}")
    print(f"  Other:         {stats['other_error']}")
    return SimpleDataset(data_list), stats


# ===========================================================
# Self-Supervised Pretraining Wrapper
# ===========================================================

class Conformer3DPretrainer(nn.Module):
    """
    Pretrains ONLY the 3D part of NMR3DNet using:
      - Task 1: Edge distance prediction (raw Angstroms)
      - Task 2: Coordinate denoising (raw Angstroms)
    """

    def __init__(self, base_model: NMR3DNet):
        super().__init__()
        self.base = base_model

        # 2D backbone (used for context; frozen during pretraining)
        self.node_proj_2d = self.base.node_proj_2d
        self.nnconv_2d = self.base.nnconv_2d
        self.gru = self.base.gru
        self.num_steps = self.base.num_step_message_passing
        self.F = self.base.node_hidden

        # 3D modules (to be trained)
        self.edge_proj_3d = self.base.edge_proj_3d
        self.rbf_encoder = getattr(self.base, "rbf_encoder", None)
        self.schnet_layers = getattr(self.base, "schnet_layers", None)
        self.egnn_layers = getattr(self.base, "egnn_layers", None)
        self.fusion_gate_3d = getattr(self.base, "fusion_gate_3d", None)
        self.fusion_gate_2d3d = getattr(self.base, "fusion_gate_2d3d", None)
        self.comenet_node_proj = getattr(self.base, "comenet_node_proj", None)
        self.use_comenet = getattr(self.base, "use_comenet", False)
        self.comenet_K = getattr(self.base, "comenet_K", 4)

        # Self-supervised task heads
        self.distance_head = nn.Sequential(
            nn.Linear(self.F * 2, self.F),
            nn.ReLU(),
            nn.Linear(self.F, 1)
        )
        self.denoising_head = nn.Sequential(
            nn.Linear(self.F, self.F),
            nn.ReLU(),
            nn.Linear(self.F, 3)
        )

    @torch.no_grad()
    def _compute_angle_embed(self, data):
        """Optional ComENet-lite per-node angles"""
        if not (self.use_comenet and hasattr(data, 'pos')):
            return None
        ang = compute_node_angle_features(data, K=self.comenet_K)
        if self.comenet_node_proj is None:
            return None
        return self.comenet_node_proj(ang)

    def encode_3d(self, data: Data):
        """
        Encode nodes with 2D backbone, then apply 3D heads and fuse.
        Returns: x3d [N, F], distances [E]
        """
        # 2D backbone
        h = self.node_proj_2d(data.x)
        for _ in range(self.num_steps):
            m2d = self.nnconv_2d(h, data.edge_index, data.edge_attr)
            _, h_next = self.gru(m2d.unsqueeze(0), h.unsqueeze(0))
            h = h_next.squeeze(0)

        # 3D edge features: projection + RBF
        src, dst = data.edge_index
        edge_lin = self.edge_proj_3d(data.edge_attr)
        distances = torch.norm(data.pos[src] - data.pos[dst], dim=-1)

        if self.rbf_encoder is not None:
            rbf = self.rbf_encoder(distances)
            edge_feat_3d = torch.cat([edge_lin, rbf], dim=-1)
        else:
            edge_feat_3d = edge_lin

        # Optional angular features
        angle_embed = self._compute_angle_embed(data)

        # SchNet / EGNN heads
        x_schnet = x_egnn = None

        if self.schnet_layers is not None:
            x_schnet = h
            for layer in self.schnet_layers:
                x_schnet = layer(x_schnet, data.edge_index, edge_feat_3d)
            if angle_embed is not None:
                x_schnet = x_schnet + angle_embed

        if self.egnn_layers is not None:
            x_egnn = h
            for layer in self.egnn_layers:
                x_egnn = layer(x_egnn, data.pos, data.edge_index, edge_feat_3d)
            if angle_embed is not None:
                x_egnn = x_egnn + angle_embed

        # Fuse SchNet/EGNN
        if (x_schnet is not None) and (x_egnn is not None):
            gate_val = self.fusion_gate_3d(torch.cat([x_schnet, x_egnn], dim=-1))
            x3d = gate_val * x_schnet + (1.0 - gate_val) * x_egnn
        elif x_schnet is not None:
            x3d = x_schnet
        else:
            x3d = x_egnn

        # Fuse 2D vs 3D
        if self.fusion_gate_2d3d is not None:
            gate = self.fusion_gate_2d3d(torch.cat([h, x3d], dim=-1))
            x3d = gate * x3d + (1.0 - gate) * h

        return x3d, distances

    def task_distance(self, data: Data):
        """Task 1: Predict inter-atomic distances (raw Angstroms)"""
        x3d, true_distances = self.encode_3d(data)
        src, dst = data.edge_index
        pair = torch.cat([x3d[src], x3d[dst]], dim=-1)
        pred = self.distance_head(pair).squeeze(-1)
        return F.mse_loss(pred, true_distances)

    def task_denoise(self, data: Data, noise_level=0.10):
        """Task 2: Predict noise added to coordinates (raw Angstroms)"""
        noise = torch.randn_like(data.pos) * noise_level
        data_noisy = data.clone()
        data_noisy.pos = data.pos + noise
        x3d, _ = self.encode_3d(data_noisy)
        pred = self.denoising_head(x3d)
        return F.mse_loss(pred, noise)


# ===========================================================
# Training Utilities
# ===========================================================

def collect_3d_parameters(model: NMR3DNet, include_comenet=True):
    """Return an iterator over ONLY 3D parameters"""
    modules = []

    # 3D components
    modules.append(model.edge_proj_3d)
    if hasattr(model, "rbf_encoder") and model.rbf_encoder is not None:
        modules.append(model.rbf_encoder)
    if hasattr(model, "schnet_layers") and model.schnet_layers is not None:
        modules.append(model.schnet_layers)
    if hasattr(model, "egnn_layers") and model.egnn_layers is not None:
        modules.append(model.egnn_layers)
    if hasattr(model, "fusion_gate_3d") and model.fusion_gate_3d is not None:
        modules.append(model.fusion_gate_3d)
    if hasattr(model, "fusion_gate_2d3d") and model.fusion_gate_2d3d is not None:
        modules.append(model.fusion_gate_2d3d)
    if include_comenet and hasattr(model, "comenet_node_proj") and model.comenet_node_proj is not None:
        modules.append(model.comenet_node_proj)

    for m in modules:
        for p in m.parameters():
            yield p


def freeze_2d_backbone(model: NMR3DNet, freeze=True):
    """Freeze/unfreeze the 2D backbone"""
    for m in [model.node_proj_2d, model.nnconv_2d, model.gru]:
        for p in m.parameters():
            p.requires_grad = not freeze


def pretrain_epoch(pretrainer, loader, optimizer, device,
                   distance_weight=1.0, denoise_weight=0.5, noise_level=0.10):
    """Train for one epoch"""
    pretrainer.train()
    tot, tdist, tden = 0.0, 0.0, 0.0

    for batch in loader:
        batch = batch.to(device)

        # Two self-supervised tasks
        loss_d = pretrainer.task_distance(batch)
        loss_z = pretrainer.task_denoise(batch, noise_level=noise_level)
        loss = distance_weight * loss_d + denoise_weight * loss_z

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), 5.0)
        optimizer.step()

        tot += float(loss.item())
        tdist += float(loss_d.item())
        tden += float(loss_z.item())

    n = len(loader)
    return tot / n, tdist / n, tden / n


# ===========================================================
# Main Pretraining Function
# ===========================================================

def main():
    """
    Main pretraining function - called by pipeline runner.
    Uses centralized config from C_NMR_config.py
    """
    print(f"\n{'=' * 80}")
    print(f"SELF-SUPERVISED 3D ENCODER PRETRAINING")
    print(f"{'=' * 80}\n")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Ensure output directory exists
    os.makedirs(config.DIR_PRETRAIN, exist_ok=True)

    # Load conformer data
    dataset, load_stats = load_conformer_pickles(
        str(config.CONF_DIR_PRETRAIN),
        node_dim=config.MODEL_3D_CONFIG["node_in_dim"],
        edge_dim=config.MODEL_3D_CONFIG["edge_in_dim"]
    )

    if len(dataset) == 0:
        raise ValueError("No conformers loaded - check your pickle files!")

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=config.PRETRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    print(f"\nBatches per epoch: {len(loader)}")

    # Build base model
    base = NMR3DNet(**config.MODEL_3D_CONFIG).to(device)

    # Freeze 2D backbone
    freeze_2d_backbone(base, freeze=config.PRETRAIN_FREEZE_2D)

    # Wrap in pretrainer
    pretrainer = Conformer3DPretrainer(base).to(device)

    # Optimizer over 3D parameters only
    params_3d = list(collect_3d_parameters(base, include_comenet=config.MODEL_3D_CONFIG["use_comenet"]))
    optimizer = torch.optim.Adam(
        params_3d,
        lr=config.PRETRAIN_LR,
        weight_decay=config.PRETRAIN_WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-6
    )

    # Print configuration
    print(f"\n{'=' * 80}")
    print(f"PRETRAINING CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"  Mode: {config.PRETRAIN_MODE}")
    print(f"  Epochs: {config.PRETRAIN_EPOCHS}")
    print(f"  Batch Size: {config.PRETRAIN_BATCH_SIZE}")
    print(f"  Learning Rate: {config.PRETRAIN_LR}")
    print(f"  Distance Weight: {config.SSL_DISTANCE_WEIGHT}")
    print(f"  Denoise Weight: {config.SSL_DENOISE_WEIGHT}")
    print(f"  Noise Level: {config.SSL_NOISE_LEVEL} Å")
    print(f"  Freeze 2D Backbone: {config.PRETRAIN_FREEZE_2D}")
    print(f"  Trainable Parameters: {sum(p.numel() for p in params_3d):,}")
    print(f"{'=' * 80}\n")

    # Training loop
    print(f"{'=' * 80}")
    print(f"TRAINING 3D ENCODER")
    print(f"{'=' * 80}\n")

    best = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(1, config.PRETRAIN_EPOCHS + 1):
        loss, ldist, lden = pretrain_epoch(
            pretrainer, loader, optimizer, device,
            distance_weight=config.SSL_DISTANCE_WEIGHT,
            denoise_weight=config.SSL_DENOISE_WEIGHT,
            noise_level=config.SSL_NOISE_LEVEL
        )
        scheduler.step(loss)

        history.append({
            'epoch': epoch,
            'loss': loss,
            'dist_loss': ldist,
            'denoise_loss': lden,
            'lr': optimizer.param_groups[0]['lr']
        })

        # Print every epoch
        print(f"Epoch {epoch:3d}: loss={loss:.4f} (dist={ldist:.4f}, denoise={lden:.4f}) "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if loss < best:
            best = loss
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': base.state_dict(),
                'pretrainer_state_dict': pretrainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best,
                'task': 'self_supervised',
                'config': config.MODEL_3D_CONFIG,
                'load_stats': load_stats,
                'timestamp': datetime.now().isoformat()
            }
            torch.save(checkpoint, config.PRETRAINED_PATH)
        else:
            patience_counter += 1
            if patience_counter >= config.PRETRAIN_EARLY_PATIENCE:
                print(f"\n  → Early stopping at epoch {epoch}")
                break

    # Save training history
    hist_path = config.DIR_PRETRAIN / "pretraining_history.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Final summary
    print(f"\n{'=' * 80}")
    print(f"PRETRAINING COMPLETE!")
    print(f"{'=' * 80}")
    print(f"  Best Loss: {best:.4f}")
    print(f"  Epochs Trained: {len(history)}")
    print(f"  ✓ Model saved: {config.PRETRAINED_PATH}")
    print(f"  ✓ History saved: {hist_path}")
    print(f"\nWhat the model learned:")
    print(f"  ✓ 3D geometric relationships (distance prediction)")
    print(f"  ✓ Robust structural representations (coordinate denoising)")
    print(f"  ✓ Features optimized for raw ppm prediction")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()