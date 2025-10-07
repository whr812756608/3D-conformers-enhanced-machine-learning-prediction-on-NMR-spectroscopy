# GEOM_pretraining.py
"""
Self-supervised pretraining to specialize the 3D part of NMR3DNet
(2D backbone is reused/frozen; 3D heads are trained on conformers)
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

from data_competition_1d_code_final_ver.Model_3D_NMR_hybrid import NMR3DNet, compute_node_angle_features
torch.manual_seed(42)
np.random.seed(42)


# ------------------------------ Data utils ------------------------------

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
    stats = {'attempted': 0, 'success': 0, 'no_conformers': 0, 'no_mol': 0, 'coord_error': 0, 'other_error': 0}

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

            # Node features (toy; keep dims stable)
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

            edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # [2, E]
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)     # [E, edge_dim]

            # 3D coordinates with canonical ordering
            non_h_indices = [i for i, a in enumerate(rd_mol.GetAtoms()) if a.GetAtomicNum() != 1]
            all_coords = rd_mol.GetConformer(0).GetPositions()
            pos_coords = all_coords[non_h_indices]
            canonical_order = canonical_atom_order(rd_mol)
            pos_reordered = pos_coords[canonical_order]
            pos = torch.tensor(pos_reordered, dtype=torch.float32)       # [N, 3]

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


# ------------------------------ Pretrainer ------------------------------

class Conformer3DPretrainer(nn.Module):
    """
    Pretrains ONLY the 3D part of NMR3DNet using:
      - Task 1: Edge distance prediction
      - Task 2: Coordinate denoising (predict Δpos per atom)
    """
    def __init__(self, base_model: NMR3DNet):
        super().__init__()
        self.base = base_model

        # 2D backbone (used for context; typically frozen)
        self.node_proj_2d = self.base.node_proj_2d
        self.nnconv_2d    = self.base.nnconv_2d
        self.gru          = self.base.gru
        self.num_steps    = self.base.num_step_message_passing
        self.F            = self.base.node_hidden

        # 3D modules (to be trained)
        self.edge_proj_3d   = self.base.edge_proj_3d
        self.rbf_encoder    = getattr(self.base, "rbf_encoder", None)
        self.schnet_layers  = getattr(self.base, "schnet_layers", None)
        self.egnn_layers    = getattr(self.base, "egnn_layers", None)
        self.fusion_gate_3d = getattr(self.base, "fusion_gate_3d", None)
        self.fusion_gate_2d3d = getattr(self.base, "fusion_gate_2d3d", None)
        self.comenet_node_proj = getattr(self.base, "comenet_node_proj", None)
        self.use_comenet    = getattr(self.base, "use_comenet", False)
        self.comenet_K      = getattr(self.base, "comenet_K", 4)

        # SSL heads
        self.distance_head = nn.Sequential(
            nn.Linear(self.F * 2, self.F), nn.ReLU(),
            nn.Linear(self.F, 1)
        )
        self.denoising_head = nn.Sequential(
            nn.Linear(self.F, self.F), nn.ReLU(),
            nn.Linear(self.F, 3)
        )

    @torch.no_grad()
    def _compute_angle_embed(self, data):
        """Optional ComENet-lite per-node angles -> [N, F] (uses the base's proj)"""
        if not (self.use_comenet and hasattr(data, 'pos')):
            return None
        ang = compute_node_angle_features(data, K=self.comenet_K)  # [N, 2K+2]
        if self.comenet_node_proj is None:
            return None
        return self.comenet_node_proj(ang)  # [N, F]

    def encode_3d(self, data: Data):
        """
        Encode nodes with 2D backbone, then apply 3D heads and fuse to x3d.
        Returns: x3d [N, F], distances [E]
        """
        # --- 2D backbone ---
        h = self.node_proj_2d(data.x)
        for _ in range(self.num_steps):
            m2d = self.nnconv_2d(h, data.edge_index, data.edge_attr)
            _, h_next = self.gru(m2d.unsqueeze(0), h.unsqueeze(0))
            h = h_next.squeeze(0)

        # --- 3D edge features: proj + RBF ---
        src, dst = data.edge_index
        edge_lin = self.edge_proj_3d(data.edge_attr)               # [E, hidden]
        distances = torch.norm(data.pos[src] - data.pos[dst], dim=-1)
        if self.rbf_encoder is not None:
            rbf = self.rbf_encoder(distances)                      # [E, n_rbf]
            edge_feat_3d = torch.cat([edge_lin, rbf], dim=-1)      # [E, hidden+n_rbf]
        else:
            edge_feat_3d = edge_lin

        # --- optional angular (ComENet-lite) ---
        angle_embed = self._compute_angle_embed(data)  # [N, F] or None

        # --- SchNet / EGNN heads from h ---
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

        # --- fuse SchNet/EGNN ---
        if (x_schnet is not None) and (x_egnn is not None):
            x3d = self.fusion_gate_3d(torch.cat([x_schnet, x_egnn], dim=-1)) * x_schnet \
                + (1.0 - self.fusion_gate_3d(torch.cat([x_schnet, x_egnn], dim=-1))) * x_egnn
        elif x_schnet is not None:
            x3d = x_schnet
        else:
            x3d = x_egnn

        # --- fuse 2D vs 3D final (keeps pretrain consistent with finetune) ---
        if self.fusion_gate_2d3d is not None:
            gate = self.fusion_gate_2d3d(torch.cat([h, x3d], dim=-1))
            x3d = gate * x3d + (1.0 - gate) * h

        return x3d, distances

    def task_distance(self, data: Data):
        x3d, true_d = self.encode_3d(data)
        src, dst = data.edge_index
        pair = torch.cat([x3d[src], x3d[dst]], dim=-1)  # [E, 2F]
        pred = self.distance_head(pair).squeeze(-1)
        return F.mse_loss(pred, true_d)

    def task_denoise(self, data: Data, noise_level=0.10):
        noise = torch.randn_like(data.pos) * noise_level
        data_noisy = data.clone()
        data_noisy.pos = data.pos + noise
        x3d, _ = self.encode_3d(data_noisy)
        pred = self.denoising_head(x3d)  # [N, 3]
        return F.mse_loss(pred, noise)


# ------------------------------ Train utils ------------------------------

def collect_3d_parameters(model: NMR3DNet, include_comenet=True):
    """Return an iterator over ONLY 3D parameters."""
    modules = []
    # 3D edges / filters
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
    """Freeze/unfreeze the 2D backbone (node_proj_2d, nnconv_2d, gru)."""
    for m in [model.node_proj_2d, model.nnconv_2d, model.gru]:
        for p in m.parameters():
            p.requires_grad = not freeze


def pretrain_epoch(pretrainer, loader, optimizer, device, denoise_weight=0.5):
    pretrainer.train()
    tot, tdist, tden = 0.0, 0.0, 0.0
    for batch in loader:
        batch = batch.to(device)
        loss_d = pretrainer.task_distance(batch)
        loss_z = pretrainer.task_denoise(batch, noise_level=0.1)
        loss = loss_d + denoise_weight * loss_z

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), 5.0)
        optimizer.step()

        tot += float(loss.item())
        tdist += float(loss_d.item())
        tden += float(loss_z.item())
    n = len(loader)
    return tot / n, tdist / n, tden / n


# ------------------------------ Main ------------------------------

def main():
    # ======= CONFIG =======
    CONFORMER_FOLDER = "./conformer_pickles_pretrain"
    OUTPUT_DIR = "./pretrained_models"
    TASK_NAME = "3D"

    # Architecture (must match fine-tuning)
    NODE_DIM = 74
    EDGE_DIM = 9
    NODE_HIDDEN = 64       # 2D backbone hidden
    EDGE_HIDDEN = 128      # 3D edge projection size
    N_INTERACTIONS = 4
    USE_SCHNET = True
    USE_EGNN = True
    USE_COMENET = True
    COMENET_K = 4

    # Training
    EPOCHS = 20
    BATCH_SIZE = 32
    LR = 1e-3
    DENOISE_WEIGHT = 0.5
    FREEZE_2D = True
    # =======================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Data
    dataset, load_stats = load_conformer_pickles(CONFORMER_FOLDER, NODE_DIM, EDGE_DIM)
    if len(dataset) == 0:
        raise ValueError("No conformers loaded")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"\nBatches per epoch: {len(loader)}")

    # Base model (same as finetune model; 3D enabled)
    base = NMR3DNet(
        node_in_dim=NODE_DIM, edge_in_dim=EDGE_DIM,
        node_hidden=NODE_HIDDEN, mpnn_embed=256,
        num_step_message_passing=5, set2set_steps=3,
        edge_hidden=EDGE_HIDDEN, n_interactions=N_INTERACTIONS,
        n_rbf=32, rbf_cutoff=8.0,
        use_3d=True, use_schnet=USE_SCHNET, use_egnn=USE_EGNN,
        use_comenet=USE_COMENET, comenet_K=COMENET_K, comenet_into_2d=False,
        readout_hidden=512, dropout=0.1
    ).to(device)

    # Freeze 2D backbone if requested
    freeze_2d_backbone(base, freeze=FREEZE_2D)

    # Wrap in pretrainer that uses only the encoder paths
    pretrainer = Conformer3DPretrainer(base).to(device)

    # Optimizer ONLY over 3D parameters
    params_3d = list(collect_3d_parameters(base, include_comenet=USE_COMENET))
    optimizer = torch.optim.Adam(params_3d, lr=LR, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-6
    )

    # Train
    print(f"\n{'=' * 80}\nPRETRAINING 3D ENCODER\n{'=' * 80}")
    best = float('inf')
    history = []

    for epoch in range(1, EPOCHS + 1):
        loss, ldist, lden = pretrain_epoch(pretrainer, loader, optimizer, device, DENOISE_WEIGHT)
        scheduler.step(loss)

        history.append({'epoch': epoch, 'loss': loss, 'dist_loss': ldist, 'denoise_loss': lden,
                        'lr': optimizer.param_groups[0]['lr']})
        print(f"Epoch {epoch:3d}: loss={loss:.4f} (dist={ldist:.4f}, denoise={lden:.4f}) "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        if loss < best:
            best = loss
            save_path = os.path.join(OUTPUT_DIR, f"{TASK_NAME}_pretrained.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': base.state_dict(),   # save base (to load for finetuning)
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': {
                    'node_dim': NODE_DIM, 'edge_dim': EDGE_DIM,
                    'node_hidden': NODE_HIDDEN, 'edge_hidden': EDGE_HIDDEN,
                    'n_interactions': N_INTERACTIONS,
                    'use_schnet': USE_SCHNET, 'use_egnn': USE_EGNN,
                    'use_comenet': USE_COMENET, 'comenet_k': COMENET_K,
                    'task_name': TASK_NAME
                },
                'stats': load_stats,
                'timestamp': datetime.now().isoformat()
            }, save_path)

    # Save history
    hist_path = os.path.join(OUTPUT_DIR, f"{TASK_NAME}_pretrain_history.json")
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 80}\nPRETRAINING COMPLETE\n{'=' * 80}")
    print(f"Best loss: {best:.4f}")
    print(f"✓ Model saved: {save_path}")
    print(f"✓ History saved: {hist_path}")
    print("\nFinetune load snippet:")
    print(f"  ckpt = torch.load('{save_path}', map_location='cpu')")
    print(f"  model = NMR3DNet(node_in_dim={NODE_DIM}, edge_in_dim={EDGE_DIM}, "
          f"node_hidden={NODE_HIDDEN}, edge_hidden={EDGE_HIDDEN}, n_interactions={N_INTERACTIONS}, "
          f"use_3d=True, use_schnet={USE_SCHNET}, use_egnn={USE_EGNN}, "
          f"use_comenet={USE_COMENET}, comenet_K={COMENET_K})")
    print("  model.load_state_dict(ckpt['model_state_dict'])")


if __name__ == "__main__":
    main()
