# Pretrain_GEOM_model.py
# Geometry-aware self-supervised pretraining for your hybrid NMR GNN.
# JSON path (as requested) is hardcoded below.

import os, json, math, random, pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, Batch
#from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader  # use the vanilla one

from rdkit import Chem


from data_competition_1d_code_data.Model_3D_NMR_hybrid import (
    GaussianBasis,        # RBF encoder
    SchNetInteraction,    # SchNet interaction layer
    EGNNLayer,            # EGNN layer
)
from torch_geometric.nn import Set2Set  # also used in your model

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
JSON_PATH = r"E:\GEOM\data_competition_1d_code_data\GEOM_pretrain\geom_stability_dataset_5conf.json"

cfg = dict(
    # Backbone dims aligned to what we BUILD from RDKit (node_in=1, edge_in=4):
    node_in_dim=1,
    edge_in_dim=4,
    node_hidden=128,
    edge_hidden=256,
    n_interactions=4,
    n_rbf=32,
    rbf_cutoff=8.0,
    readout_hidden=256,    # not used by SSL head, but we keep a tidy head for completeness
    dropout=0.1,

    # Training
    batch_size=16,
    lr=3e-4,
    weight_decay=1e-5,
    epochs=30,
    warmup_steps=500,
    max_steps=20000,
    grad_clip=1.0,
    device="cuda" if torch.cuda.is_available() else "cpu",

    # SSL losses
    rbf_bins=64,
    rbf_cutoff_ssl=5.0,
    w_rbf=1.0,
    w_mask=0.5,

    # Masking
    mask_ratio=0.15,

    # Conformers per molecule
    max_confs_per_mol=5,

    # Save
    save_dir="./pretrain_ckpts",
    backbone_ckpt="pretrained_backbone.pt",
)

Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
BOND_ORDER_MAP = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}

def _mol_to_graph_tensors(mol_no_h: Chem.Mol):
    """Return (x, edge_index, edge_attr) for heavy-atom RDKit mol."""
    N = mol_no_h.GetNumAtoms()
    if N == 0:
        return None

    Z = [a.GetAtomicNum() for a in mol_no_h.GetAtoms()]   # scalar feature
    x = torch.tensor(np.array(Z, dtype=np.float32)).view(N, 1)  # [N,1]

    src, dst, eattr = [], [], []
    for b in mol_no_h.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bo = BOND_ORDER_MAP.get(b.GetBondType(), 0)
        oh = [0, 0, 0, 0]; oh[bo] = 1
        src.extend([i, j]); dst.extend([j, i])
        eattr.extend([oh, oh])

    if len(src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 4), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(np.array(eattr, dtype=np.float32))
    return x, edge_index, edge_attr

def _best_n_conformers(rec_dict, n_take: int):
    confs = rec_dict.get("conformers", [])
    if not isinstance(confs, list) or len(confs) == 0:
        return []
    return sorted(confs, key=lambda c: c.get("boltzmannweight", 0.0), reverse=True)[:n_take]

def edge_lengths(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    s, t = edge_index
    vec = pos[s] - pos[t]
    return torch.linalg.norm(vec, dim=-1)

def rbf_basis(dists: torch.Tensor, K: int, cutoff: float) -> torch.Tensor:
    device = dists.device
    centers = torch.linspace(0.0, cutoff, K, device=device)
    gamma = 1.0 / ((cutoff / K) ** 2 + 1e-9)
    return torch.exp(-gamma * (dists.unsqueeze(-1) - centers) ** 2)

def mask_node_features(x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    N = x.size(0)
    k = max(1, int(round(mask_ratio * N)))
    idx = torch.randperm(N, device=x.device)[:k]
    mask = torch.zeros(N, dtype=torch.bool, device=x.device); mask[idx] = True
    x_masked = x.clone(); x_masked[mask] = 0.0
    return x_masked, mask

# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class JSONPretrainDataset(torch.utils.data.Dataset):
    """Each item: list of conformers turned into PyG Data objects (we’ll flatten in collate)."""
    def __init__(self, json_path: str, max_confs_per_mol: Optional[int] = None, verbose: bool = True):
        self.json_path = json_path
        print("Loading JSON pretrain dataset ...")
        with open(json_path, "r") as f:
            self.rows = json.load(f)
        self.max_confs_per_mol = max_confs_per_mol
        self.verbose = verbose
        print(f"Loaded {len(self.rows)} molecules from {json_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        pkl_path = row["pickle_path"]
        n_take = self.max_confs_per_mol or int(row.get("n_conformers_used", 5))

        try:
            with open(pkl_path, "rb") as f:
                rec = pickle.load(f)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] pickle load failed: {pkl_path} | {e}")
            return []  # no conformers

        confs = _best_n_conformers(rec, n_take)
        data_list = []

        for c in confs:
            rd_mol = c.get("rd_mol")
            if rd_mol is None or rd_mol.GetNumConformers() == 0:
                continue
            try:
                mol_no_h = Chem.RemoveHs(rd_mol, sanitize=False)
            except Exception:
                try:
                    Chem.SanitizeMol(rd_mol)
                    mol_no_h = Chem.RemoveHs(rd_mol, sanitize=False)
                except Exception:
                    continue
            if mol_no_h is None or mol_no_h.GetNumAtoms() == 0:
                continue

            pos = torch.tensor(mol_no_h.GetConformer(0).GetPositions(), dtype=torch.float32)
            g = _mol_to_graph_tensors(mol_no_h)
            if g is None:
                continue
            x, edge_index, edge_attr = g

            # mask: SSL uses all atoms; set True
            mask = torch.ones(x.size(0), dtype=torch.bool)

            data = Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, mask=mask
            )
            data_list.append(data)

        return data_list

def pretrain_collate(batch):
    # batch is List[List[Data]]; flatten
    flattened: List[Data] = []
    for conf_list in batch:
        flattened.extend(conf_list)
    if len(flattened) == 0:
        # return an empty placeholder Batch; train loop will skip
        return None
    return Batch.from_data_list(flattened)

# --------------------------------------------------------------------------------------
# Backbone with feature access (mirrors your architecture; returns node embeddings)
# --------------------------------------------------------------------------------------
class NMRBackbone(nn.Module):
    """
    Feature-only backbone that mirrors your NMR3DNet blocks, but exposes
    per-node embeddings (x_out) via forward_features(...) for SSL.
    Uses node_in_dim=1 and edge_in_dim=4 for the pretraining graphs we build.
    """
    def __init__(
        self,
        node_in_dim=1,
        edge_in_dim=4,
        node_hidden=128,
        edge_hidden=256,
        n_interactions=4,
        n_rbf=32,
        rbf_cutoff=8.0,
        use_schnet=True,
        use_egnn=True,
        use_3d=True,  # during SSL we DO use 3D (pos) for RBF edge head
    ):
        super().__init__()
        self.use_schnet = use_schnet
        self.use_egnn = use_egnn
        self.use_3d    = use_3d
        self.n_interactions = n_interactions

        self.node_proj = nn.Linear(node_in_dim, node_hidden)
        self.edge_proj = nn.Linear(edge_in_dim, edge_hidden)

        if self.use_3d:
            self.rbf_encoder = GaussianBasis(0.0, rbf_cutoff, n_rbf)
            edge_dim_with_3d = edge_hidden + n_rbf
        else:
            edge_dim_with_3d = edge_hidden

        if self.use_schnet:
            self.schnet_layers = nn.ModuleList([
                SchNetInteraction(node_hidden, edge_dim_with_3d, edge_hidden)
                for _ in range(n_interactions)
            ])

        if self.use_egnn and self.use_3d:
            self.egnn_layers = nn.ModuleList([
                EGNNLayer(node_hidden, edge_dim_with_3d, edge_hidden)
                for _ in range(n_interactions)
            ])

        # Fuse if both paths active (same as your model)  :contentReference[oaicite:3]{index=3}
        if self.use_schnet and self.use_egnn and self.use_3d:
            self.fusion_gate = nn.Sequential(
                nn.Linear(node_hidden * 2, node_hidden),
                nn.Sigmoid()
            )

    def forward_features(self, data: Data) -> torch.Tensor:
        x = self.node_proj(data.x)
        edge_features = self.edge_proj(data.edge_attr)

        if self.use_3d and hasattr(data, 'pos'):
            s, t = data.edge_index
            distances = torch.norm(data.pos[s] - data.pos[t], dim=-1)
            rbf_features = self.rbf_encoder(distances)
            edge_features_3d = torch.cat([edge_features, rbf_features], dim=-1)
        else:
            edge_features_3d = edge_features

        x_out = x

        if self.use_schnet:
            x_schnet = x
            for i in range(self.n_interactions):
                x_schnet = self.schnet_layers[i](x_schnet, data.edge_index, edge_features_3d)
            x_out = x_schnet

        if self.use_egnn and self.use_3d and hasattr(data, 'pos'):
            x_egnn = x
            for i in range(self.n_interactions):
                x_egnn = self.egnn_layers[i](x_egnn, data.pos, data.edge_index, edge_features_3d)

            if self.use_schnet:
                gate = self.fusion_gate(torch.cat([x_schnet, x_egnn], dim=-1))
                x_out = gate * x_schnet + (1 - gate) * x_egnn
            else:
                x_out = x_egnn

        return x_out  # [sumN, node_hidden]

# --------------------------------------------------------------------------------------
# SSL heads
# --------------------------------------------------------------------------------------
class RBFPredictor(nn.Module):
    def __init__(self, node_dim: int, rbf_bins: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, rbf_bins)
        )
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        s, t = edge_index
        return self.mlp(torch.cat([h[s], h[t]], dim=-1))

class MaskDecoder(nn.Module):
    def __init__(self, node_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, out_dim)
        )
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)

# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------
def ssl_losses(backbone: NMRBackbone,
               rbf_head: RBFPredictor,
               mask_head: MaskDecoder,
               batch: Batch,
               cfg: Dict):
    device = cfg["device"]
    data = batch.to(device)
    # targets
    x0 = data.x.float()

    # mask some nodes
    x_masked, mask_nodes = mask_node_features(x0, cfg["mask_ratio"])
    data.x = x_masked

    # per-node embeddings
    h = backbone.forward_features(data)  # [sumN, D]

    # (1) edge distance RBF
    d = edge_lengths(data.pos, data.edge_index)
    with torch.no_grad():
        rbf_tgt = rbf_basis(d, cfg["rbf_bins"], cfg["rbf_cutoff_ssl"])  # [E,K]
    rbf_logits = rbf_head(h, data.edge_index)
    loss_rbf = F.mse_loss(torch.sigmoid(rbf_logits), rbf_tgt)

    # (2) masked node feature reconstruction (rebuild atomic number scalar)
    x_pred = mask_head(h)  # [sumN, 1]
    if mask_nodes.any():
        loss_mask = F.l1_loss(x_pred[mask_nodes], x0[mask_nodes])
    else:
        loss_mask = torch.tensor(0.0, device=device)

    loss = cfg["w_rbf"] * loss_rbf + cfg["w_mask"] * loss_mask
    logs = dict(loss=loss.item(), loss_rbf=loss_rbf.item(), loss_mask=loss_mask.item())
    return loss, logs

# --------------------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------------------
def train():
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    device = torch.device(cfg["device"])

    ds = JSONPretrainDataset(JSON_PATH, max_confs_per_mol=cfg["max_confs_per_mol"], verbose=True)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
                        collate_fn=pretrain_collate)

    # model + heads
    backbone = NMRBackbone(
        node_in_dim=cfg["node_in_dim"],
        edge_in_dim=cfg["edge_in_dim"],
        node_hidden=cfg["node_hidden"],
        edge_hidden=cfg["edge_hidden"],
        n_interactions=cfg["n_interactions"],
        n_rbf=cfg["n_rbf"],
        rbf_cutoff=cfg["rbf_cutoff"],
        use_schnet=True,
        use_egnn=True,
        use_3d=True,
    ).to(device)

    rbf_head  = RBFPredictor(cfg["node_hidden"], cfg["rbf_bins"]).to(device)
    mask_head = MaskDecoder(cfg["node_hidden"], out_dim=cfg["node_in_dim"]).to(device)

    params = list(backbone.parameters()) + list(rbf_head.parameters()) + list(mask_head.parameters())
    optim = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # cosine w/ warmup
    est_steps = cfg["epochs"] * max(1, len(loader))
    def lr_lambda(step):
        if step < cfg["warmup_steps"]:
            return float(step + 1) / float(cfg["warmup_steps"])
        progress = (step - cfg["warmup_steps"]) / max(1, (est_steps - cfg["warmup_steps"]))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_running = float("inf"); step = 0

    for epoch in range(1, cfg["epochs"] + 1):
        backbone.train(); rbf_head.train(); mask_head.train()
        running = 0.0; nb = 0

        for batch in loader:
            if batch is None:  # all empties
                continue
            step += 1
            if step > cfg["max_steps"]:
                break

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss, logs = ssl_losses(backbone, rbf_head, mask_head, batch, cfg)
            scaler.scale(loss).backward()
            if cfg["grad_clip"] and cfg["grad_clip"] > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            scaler.step(optim); scaler.update(); sched.step()

            running += logs["loss"]; nb += 1
            if step % 100 == 0:
                print(f"[epoch {epoch:02d} step {step:05d}] "
                      f"ssl={logs['loss']:.4f} | rbf={logs['loss_rbf']:.4f} | mask={logs['loss_mask']:.4f}")

        if nb > 0:
            running /= nb
            print(f"[epoch {epoch:02d}] train_ssl={running:.4f}")
            if running < best_running:
                best_running = running
                ckpt_path = Path(cfg["save_dir"]) / cfg["backbone_ckpt"]
                torch.save({
                    "backbone": backbone.state_dict(),
                    "cfg": cfg,
                    "note": "SSL (edge-RBF + masked-node) pretraining backbone only"
                }, ckpt_path)
                print(f"  -> saved best backbone: {ckpt_path} (ssl={best_running:.4f})")

    print("Done.")

if __name__ == "__main__":
    train()
