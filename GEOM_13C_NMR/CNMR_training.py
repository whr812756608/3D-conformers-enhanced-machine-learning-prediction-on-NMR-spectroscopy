# train_C_from_splits.py — raw L1 loss (no normalization), with param & epoch stats
import os
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset, Subset
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error
from GEOM_models.Model_2D_3D_hybrid import NMR2DMPNN, NMR3DNet, evaluate
from pathlib import Path
import sys
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_PT       = REPO_ROOT / "GEOM_1D_graph_datasets" / "C_dataset_list.pt"
SPLIT_NPZ        = REPO_ROOT / "GEOM_1D_graph_datasets" / "C_split_indices.npz"
PRETRAINED_PATH  = REPO_ROOT / "GEOM_pretrain" / "pretrained_models" / "3D_pretrained.pt"
SAVE_DIR         = REPO_ROOT / "GEOM_models"

IF_PRETRAIN      = True
EPOCHS           = 100
BATCH_SIZE       = 32
LR               = 5e-4
WEIGHT_DECAY     = 1e-6
EARLY_PATIENCE   = 20
SEEDS            = [1, 24, 66]

PRINT_DEVICE     = True
PRINT_SPLIT_INFO = True
NUM_WORKERS      = 0
PIN_MEMORY       = False  # set False on CPU-only machines
# ===========================================================

class SimpleDataset(TorchDataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

def build_2d():
    return NMR2DMPNN(
        node_feats=64, embed_feats=256,
        num_step_message_passing=5, num_step_set2set=3,
        hidden_feats=512, prob_dropout=0.1
    )

def build_3d():
    return NMR3DNet(
        node_hidden=64, edge_hidden=128, n_interactions=4,
        use_3d=True, use_schnet=True, use_egnn=True, use_comenet=True, comenet_K=4
    )

def maybe_load_pretrained(model, pretrained_path, device):
    if not pretrained_path or not os.path.exists(pretrained_path):
        print(f"Pretrained path not found: {pretrained_path}")
        return False
    ckpt = torch.load(pretrained_path, map_location=device)
    try:
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path} (strict=False)")
        return True
    except Exception as e:
        print(f"Failed to load pretrained weights: {e}")
        return False

# ---------- RAW L1 LOSS (no normalization) ----------
def train_epoch(model, loader, optimizer, device, lambda_reg=0.02, max_grad_norm=5.0):
    model.train()
    total_loss = total_pred_loss = total_reg_loss = 0.0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        target = data.y[data.mask]  # raw target

        pred_loss = torch.nn.functional.l1_loss(pred, target)  # raw L1
        loss = pred_loss

        # optional regularization (no-op unless your model has selector/use_adaptive)
        if hasattr(model, 'selector') and getattr(model, 'use_adaptive', False):
            weights = model.selector(data.x)
            reg_loss = lambda_reg * (0.65 - weights.mean()).clamp(min=0) ** 2
            loss = loss + reg_loss
            total_reg_loss += reg_loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        total_pred_loss += pred_loss.item()

    n = max(1, len(loader))
    avg_reg = (total_reg_loss / n) if total_reg_loss > 0 else 0.0
    return total_loss / n, total_pred_loss / n, avg_reg

@torch.no_grad()
def eval_mae(model, loader, device):
    """Evaluation wrapper that temporarily switches to eval() and disables grad."""
    was_training = model.training
    model.eval()
    pred, true = evaluate(model, loader, device)
    if was_training:
        model.train()
    return mean_absolute_error(true, pred)

def train_model(config_name, model, train_set, val_set, device,
                epochs=100, bs=32, lr=5e-4, weight_decay=1e-6,
                lambda_reg=0.02, early_stop_patience=20):

    print(f"\n{'='*80}\nTraining: {config_name}\n{'='*80}")

    # loaders (perf-tuned; adjust NUM_WORKERS/PIN_MEMORY if needed)
    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=(NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_set, batch_size=bs, shuffle=False, drop_last=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=(NUM_WORKERS > 0)
    )

    # parameter count
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_mae = float('inf')
    patience = 0
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{config_name}.pt")

    for epoch in range(1, epochs + 1):
        train_loss, pred_loss, reg_loss = train_epoch(
            model, train_loader, optimizer, device, lambda_reg=lambda_reg
        )

        # validation for early stopping / scheduler
        val_mae = eval_mae(model, val_loader, device)
        scheduler.step(val_mae)

        # epoch summary
        if (epoch % 20) == 0:
            if reg_loss > 0:
                print(f"  Epoch {epoch:3d}: loss={train_loss:.4f} (pred={pred_loss:.4f}, reg={reg_loss:.5f}), val={val_mae:.4f}")
            else:
                print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}, val={val_mae:.4f}")

        # checkpoint on best val
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), save_path)
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"  Early stop at epoch {epoch}")
                break

    print(f"Saved best checkpoint -> {save_path}")
    return save_path, best_val_mae

def main():
    # ---- global determinism ----
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if PRINT_DEVICE:
        print(f"Using device: {device}")

    # dataset & splits
    if not (os.path.exists(DATASET_PT) and os.path.exists(SPLIT_NPZ)):
        raise FileNotFoundError(
            f"Missing dataset/splits.\nExpected:\n  {DATASET_PT}\n  {SPLIT_NPZ}\n"
            "Make sure your working directory is the repo root and that you ran make_splits_C.py."
        )

    data_list = torch.load(DATASET_PT, map_location="cpu")
    dataset = SimpleDataset(data_list)
    split = np.load(SPLIT_NPZ)
    train_idx = split["train_idx"]; val_idx = split["val_idx"]; test_idx = split["test_idx"]

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)

    if PRINT_SPLIT_INFO:
        print(f"Split sizes: train={len(train_set)}, val={len(val_set)}, test={len(test_idx)}")

    for SEED in SEEDS:
        print(f"\n========== SEED {SEED} ==========")
        # per-seed determinism
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        # 2D baseline
        model_2d = build_2d().to(device)
        # train_model(
        #     f"C_2D_baseline_seed{SEED}", model_2d, train_set, val_set, device,
        #     epochs=EPOCHS, bs=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY, early_stop_patience=EARLY_PATIENCE
        # )

        # 3D single conformer (from scratch)
        model_3d = build_3d().to(device)
        # train_model(
        #     f"C_3D_single_seed{SEED}", model_3d, train_set, val_set, device,
        #     epochs=EPOCHS, bs=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY, early_stop_patience=EARLY_PATIENCE
        # )

        # 3D single conformer (pretrained init)
        if IF_PRETRAIN:
            model_3d_pt = build_3d().to(device)
            ok = maybe_load_pretrained(model_3d_pt, PRETRAINED_PATH, device)
            if not ok:
                print("Proceeding without pretrained (weights not loaded).")
            train_model(
                f"C_3D_single_pretrained_seed{SEED}", model_3d_pt, train_set, val_set, device,
                epochs=EPOCHS, bs=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY, early_stop_patience=EARLY_PATIENCE
            )

    print("\nAll training runs complete. Check ./GEOM_models for saved checkpoints.")

if __name__ == "__main__":
    main()
