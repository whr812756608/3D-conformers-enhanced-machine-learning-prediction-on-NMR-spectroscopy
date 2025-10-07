# HSQC_benchmark.py (memory-safe)
# Train 10 epochs and compare Baseline MAE vs strict Boltzmann-weighted Ensemble MAE
# Models: Transformer+nnconv, Pure GNN (nnconv), ComENet, SchNet

import os, json, pickle, time, warnings
from types import SimpleNamespace
from typing import List, Dict

import numpy as np
from rdkit import Chem

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.backends.cuda import sdp_kernel
from torch_geometric.data import Data

# Quiet warnings / prefer math kernels over flash attention
warnings.filterwarnings("ignore", message=r"enable_nested_tensor is True.*")
warnings.filterwarnings("ignore", message=r".*Torch was not compiled with flash attention.*")
try:
    sdp_kernel.enable_flash_sdp(False)
    sdp_kernel.enable_mem_efficient_sdp(False)
    sdp_kernel.enable_math_sdp(True)
except Exception:
    pass

from models.gnn_transformer import GNNTransformer
from models.GNN2d import GNNNodeEncoder
from models.Comenet import ComENet
from models.Schnet import SchNet
from models.NMRModel import NodeEncodeInterface
from GEOM_HSQC_loader import build_hsqc_matched_dataloaders

# ===== paths =====
CSV_PATH       = "./2DNMRGym_graph_index_map_with_geom_matched.csv"
MAPPING_REPORT = "./geom_mapping_report.csv"
GEOM_ROOT      = "./GEOM_HSQC_matched"

# ===== global config =====
SEED        = 42
DEFAULT_BS  = 32
LR          = 1e-4
N_EPOCHS    = 10
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# shared head settings (will be overridden per model when needed)
HIDDEN_CHANNELS     = 512
NUM_GNN_LAYERS      = 5
DROPOUT             = 0.30
USE_SOLVENT         = True
C_OUT_HIDDEN        = [128, 64]
H_OUT_HIDDEN        = [128, 64]
C_SOL_EMB           = 32
H_SOL_EMB           = 32

# transformer settings (defaults; overridden for trans+nnconv)
D_MODEL             = 128
NHEAD               = 4
FFN_DIM             = 512
NUM_ENCODER_LAYERS  = 4
TRANSFORMER_DROPOUT = 0.30
TRANSFORMER_ACT     = "relu"
MAX_INPUT_LEN       = 1000
POS_ENCODER         = False
NORM_INPUT          = False

# ensemble options (STRICT: use weights as-is; do not renormalize)
MAX_CONFS = None
STRICT_WEIGHTS = True  # we keep, but logic already adheres to strict sum of provided weights

# ===== model specs with per-model memory-friendly overrides =====
SPECS = [
    # Slimmed NNConv inside Transformer + smaller batch and shorter max_input_len
    # {"name": "trans+nnconv", "family": "transformer", "gnn_type": "nnconv",
    #  "overrides": {"hidden_channels": 128, "num_gnn_layers": 3, "d_model": 128,
    #                "batch_size": 8, "max_input_len": 800}},
    # Pure GNN with NNConv (still heavy) – reduce width & batch a bit
    {"name": "gnn+nnconv",   "family": "gnn", "gnn_type": "nnconv",
     "overrides": {"hidden_channels": 256, "num_gnn_layers": 4, "batch_size": 16}},
    # ComENet and SchNet usually fit at BS=32; keep defaults
    {"name": "comenet", "family": "comenet", "overrides": {"batch_size": 32}},
    {"name": "schnet",  "family": "schnet",  "overrides": {"batch_size": 32}},
]

# ================= Conformer helpers =================
def add_Hs_with_coords(rd_heavy_mol: Chem.Mol) -> Chem.Mol:
    if rd_heavy_mol.GetNumConformers() == 0:
        raise ValueError("GEOM mol has no conformer.")
    mH = Chem.AddHs(rd_heavy_mol, addCoords=True)
    _ = mH.GetConformer(0)
    return mH

def load_geom_pickle(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "conformers" in obj:
        conformers = obj["conformers"]
    elif isinstance(obj, list):
        conformers = obj
    else:
        raise ValueError(f"Unexpected pickle schema: {path}")
    out = []
    for it in conformers:
        rd_mol = it.get("rd_mol") if isinstance(it, dict) else None
        if rd_mol is None:
            continue
        try:
            _ = rd_mol.GetConformer(0)
        except Exception:
            continue
        w = it.get("boltzmannweight") if isinstance(it, dict) else None
        out.append((rd_mol, float(w) if w is not None else None))
    return out

def read_perm_graph2rd(perm_json_path: str) -> np.ndarray:
    with open(perm_json_path, "r") as f:
        mp = json.load(f)
    return np.array(mp["graph_to_rd"], dtype=np.int32)

def geom_pickle_path(geom_smiles: str) -> str:
    base = geom_smiles.strip()
    p = base if os.path.isabs(base) else os.path.join(GEOM_ROOT, base)
    root, ext = os.path.splitext(p)
    return p if ext else root + ".pickle"

# ================= DataLoaders per spec (so batch_size overrides apply) =================
def dataloaders_for_spec(spec):
    bs = spec.get("overrides", {}).get("batch_size", DEFAULT_BS)
    loaders, stats = build_hsqc_matched_dataloaders(
        csv_path=CSV_PATH,
        mapping_report_csv=MAPPING_REPORT,
        batch_size=bs,
        seed=SEED
    )
    return loaders, stats, bs

# ================= Build models (apply overrides) =================
def build_model(spec) -> NodeEncodeInterface:
    ov = spec.get("overrides", {})
    hc  = ov.get("hidden_channels", HIDDEN_CHANNELS)
    nl  = ov.get("num_gnn_layers", NUM_GNN_LAYERS)
    dmd = ov.get("d_model", D_MODEL)
    mil = ov.get("max_input_len", MAX_INPUT_LEN)

    fam = spec["family"]

    if fam == "transformer":
        args = SimpleNamespace(
            type=spec["gnn_type"],
            num_layers=nl,
            hidden_channels=hc,
            d_model=dmd,
            nhead=NHEAD,
            dim_feedforward=FFN_DIM,
            transformer_dropout=TRANSFORMER_DROPOUT,
            transformer_activation=TRANSFORMER_ACT,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            max_input_len=mil,
            transformer_norm_input=NORM_INPUT,
            pos_encoder=POS_ENCODER,
            dropout=DROPOUT,
            pretrained_gnn=None,
            freeze_gnn=None,
        )
        nodeEncoder = GNNTransformer(args)
        return NodeEncodeInterface(
            nodeEncoder,
            hidden_channels=dmd,
            c_out_hidden=C_OUT_HIDDEN,
            h_out_hidden=H_OUT_HIDDEN,
            c_solvent_emb_dim=C_SOL_EMB,
            h_solvent_emb_dim=H_SOL_EMB,
            h_out_channels=2,
            use_solvent=USE_SOLVENT,
        )

    if fam == "gnn":
        nodeEncoder = GNNNodeEncoder(
            nl, hc, JK="last", gnn_type=spec["gnn_type"], aggr="add"
        )
        return NodeEncodeInterface(
            nodeEncoder,
            hidden_channels=hc,
            c_out_hidden=C_OUT_HIDDEN,
            h_out_hidden=H_OUT_HIDDEN,
            c_solvent_emb_dim=C_SOL_EMB,
            h_solvent_emb_dim=H_SOL_EMB,
            h_out_channels=2,
            use_solvent=USE_SOLVENT,
        )

    if fam == "comenet":
        nodeEncoder = ComENet(
            in_embed_size=3, c_out_channels=1, h_out_channels=2,
            agg_method="sum",
            hidden_channels=hc,
            num_layers=nl,
            num_output_layers=2
        )
        return NodeEncodeInterface(
            nodeEncoder,
            hidden_channels=hc,
            c_out_hidden=C_OUT_HIDDEN,
            h_out_hidden=H_OUT_HIDDEN,
            c_solvent_emb_dim=C_SOL_EMB,
            h_solvent_emb_dim=H_SOL_EMB,
            h_out_channels=2,
            use_solvent=USE_SOLVENT,
        )

    if fam == "schnet":
        nodeEncoder = SchNet(
            energy_and_force=False,
            cutoff=10.0,
            num_layers=nl,
            hidden_channels=hc,
            out_channels=1,
            num_filters=256,
            num_gaussians=50
        )
        return NodeEncodeInterface(
            nodeEncoder,
            hidden_channels=hc,
            c_out_hidden=C_OUT_HIDDEN,
            h_out_hidden=H_OUT_HIDDEN,
            c_solvent_emb_dim=C_SOL_EMB,
            h_solvent_emb_dim=H_SOL_EMB,
            h_out_channels=2,
            use_solvent=USE_SOLVENT,
        )

    raise ValueError(f"Unknown family: {fam}")

# ================= Train w/ AMP (MAE) =================
def train_model_amp(model, dataloaders, optimizer, scheduler, ckpt, num_epochs=10):
    model = model.to(DEVICE)
    mae = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}\n----------")
        t0 = time.time()
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            epoch_loss = 0.0
            for batch in dataloaders[phase]:
                batch = batch.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    (c_pred, h_pred), _ = model(batch)
                    loss = mae(c_pred, batch.cnmr) + mae(h_pred, batch.hnmr)
                if phase == "train":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                epoch_loss += float(loss.detach().cpu())
            epoch_loss /= max(1, len(dataloaders[phase]))
            print(f"{phase} MAE {epoch_loss:.6f}")
            if phase == "train":
                scheduler.step()
        dt = time.time() - t0
        print("{:.0f}m {:.0f}s".format(dt // 60, dt % 60))

    print(f"Saving final model -> {ckpt}")
    torch.save(model.state_dict(), ckpt)
    model.eval()
    return model

# ================= Eval helpers (MAE) =================
MAE = nn.L1Loss(reduction="mean")

@torch.no_grad()
def eval_baseline(model, loader) -> Dict[str, float]:
    device = next(model.parameters()).device
    c_maes, h_maes = [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            (c_pred, h_pred), _ = model(batch)
        c_maes.append(float(MAE(c_pred, batch.cnmr).item()))
        h_maes.append(float(MAE(h_pred, batch.hnmr).item()))
    return {"C_MAE": float(np.mean(c_maes) if c_maes else np.nan),
            "H_MAE": float(np.mean(h_maes) if h_maes else np.nan)}

@torch.no_grad()
def eval_multiconf_ensemble(model, loader) -> Dict[str, float]:
    """
    Ensemble: re-normalize the Boltzmann weights over the subset of conformers
    that produced valid predictions. If sum_w < tiny, fall back to single-graph prediction.
    """
    device = next(model.parameters()).device
    c_maes, h_maes = [], []

    def ensure_batch_vec(g: Data, device):
        if (not hasattr(g, "batch")) or (g.batch is None):
            n = g.x.size(0) if getattr(g, "x", None) is not None else g.pos.size(0)
            g.batch = torch.zeros(n, dtype=torch.long, device=device)
        return g

    for b in loader:
        try:
            graphs: List[Data] = b.to_data_list()
        except AttributeError:
            graphs = [b]

        for g in graphs:
            g = g.to(device); g = ensure_batch_vec(g, device)

            ok = (
                hasattr(g, "perm_json") and isinstance(g.perm_json, str) and os.path.exists(g.perm_json) and
                hasattr(g, "geom_smiles") and isinstance(g.geom_smiles, str) and len(g.geom_smiles) > 0
            )
            if not ok:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    (c_pred, h_pred), _ = model(g)
                c_maes.append(float(MAE(c_pred.cpu(), g.cnmr.cpu()).item()))
                h_maes.append(float(MAE(h_pred.cpu(), g.hnmr.cpu()).item()))
                continue

            try:
                graph_to_rd = read_perm_graph2rd(g.perm_json)
                pk_path     = geom_pickle_path(g.geom_smiles)
                confs       = load_geom_pickle(pk_path) if os.path.exists(pk_path) else []
            except Exception:
                confs = []

            preds_c, preds_h, weights = [], [], []
            if confs:
                use = confs if MAX_CONFS is None else confs[:MAX_CONFS]
                for rd_mol, w in use:
                    if w is None or not np.isfinite(w):
                        continue
                    try:
                        mH = add_Hs_with_coords(rd_mol)
                        pos_rd = np.asarray(mH.GetConformer(0).GetPositions(), dtype=np.float32)
                        pos_graph = pos_rd[graph_to_rd]
                        g2 = g.clone()
                        g2.pos = torch.from_numpy(pos_graph).to(device)
                        if (not hasattr(g2, "batch")) or (g2.batch is None):
                            g2.batch = g.batch
                        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                            (c_p, h_p), _ = model(g2)
                        preds_c.append(c_p.detach().cpu().numpy())
                        preds_h.append(h_p.detach().cpu().numpy())
                        weights.append(float(w))  # source weights (already normalized across all confs)
                    except Exception:
                        continue

            # -------- RENORM over used conformers --------
            if preds_c and len(weights) == len(preds_c):
                w = np.array(weights, dtype=np.float64)
                sum_w = w.sum()
                if not np.isfinite(sum_w) or sum_w < 1e-8:
                    # Fallback to single-graph prediction
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                        (c_pred, h_pred), _ = model(g)
                    c_pred, h_pred = c_pred.detach().cpu(), h_pred.detach().cpu()
                else:
                    w = w / sum_w  # **RENORMALIZE OVER THE USED SUBSET**
                    Pc = np.stack(preds_c, axis=0)  # [P, Nc, 1]
                    Ph = np.stack(preds_h, axis=0)  # [P, Nh, 2]
                    c_pred = (w[:, None, None] * Pc).sum(axis=0)
                    h_pred = (w[:, None, None] * Ph).sum(axis=0)
                    c_pred = torch.from_numpy(c_pred)
                    h_pred = torch.from_numpy(h_pred)
            else:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    (c_pred, h_pred), _ = model(g)
                c_pred, h_pred = c_pred.detach().cpu(), h_pred.detach().cpu()

            c_maes.append(float(MAE(c_pred.cpu(), g.cnmr.cpu()).item()))
            h_maes.append(float(MAE(h_pred.cpu(), g.hnmr.cpu()).item()))

    return {"C_MAE": float(np.mean(c_maes) if c_maes else np.nan),
            "H_MAE": float(np.mean(h_maes) if h_maes else np.nan)}

# ================= Runner =================
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)

    results = []
    for spec in SPECS:
        print(f"\n==== Model: {spec['name']} ====")

        # Build dataloaders with per-spec batch size
        loaders, stats, bs = dataloaders_for_spec(spec)
        print("[HSQC] stats:", stats, f"| batch_size={bs}")

        # Build model with per-spec overrides
        model = build_model(spec)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Trainable params:", n_params)

        ckpt = f"{spec['name']}_b{bs}.pt"
        opt = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=LR)
        sch = lr_scheduler.StepLR(opt, step_size=8, gamma=0.9)

        model = train_model_amp(model, loaders, opt, sch, ckpt, num_epochs=N_EPOCHS)

        base = eval_baseline(model, loaders["test"])
        ens  = eval_multiconf_ensemble(model, loaders["test"])
        print(f"[Baseline MAE]  C={base['C_MAE']:.6f}  H={base['H_MAE']:.6f}")
        print(f"[Ensemble  MAE] C={ens['C_MAE']:.6f}  H={ens['H_MAE']:.6f}")

        results.append((spec["name"], base, ens))

        # free memory between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    print("\n==== Summary (MAE) ====")
    for name, base, ens in results:
        print(f"{name:14s} | Base C {base['C_MAE']:.6f} H {base['H_MAE']:.6f} | "
              f"Ens C {ens['C_MAE']:.6f} H {ens['H_MAE']:.6f}")

if __name__ == "__main__":
    main()