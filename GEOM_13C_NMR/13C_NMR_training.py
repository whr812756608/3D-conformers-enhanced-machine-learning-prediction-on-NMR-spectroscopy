import os
import numpy as np
import torch
import C_NMR_config as config
from torch.utils.data import Dataset as TorchDataset, Subset
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error
from GEOM_models.Model_2D_3D_hybrid import NMR2DMPNN, NMR3DNet, evaluate
import sys
import warnings
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


# ===========================================================

class SimpleDataset(TorchDataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def build_2d():
    return NMR2DMPNN(**config.MODEL_2D_CONFIG)


def build_3d():
    return NMR3DNet(**config.MODEL_3D_CONFIG)


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

def train_epoch(model, loader, optimizer, device, lambda_reg=0.02, max_grad_norm=5.0):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = total_pred_loss = total_reg_loss = 0.0

    for data in loader:
        data = data.to(device)
        pred = model(data)
        target = data.y[data.mask]

        pred_loss = torch.nn.functional.l1_loss(pred, target)
        loss = pred_loss

        # Optional regularization
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
    avg_loss = total_loss / n
    avg_pred_loss = total_pred_loss / n
    avg_reg = (total_reg_loss / n) if total_reg_loss > 0 else 0.0

    return avg_loss, avg_pred_loss, avg_reg


@torch.no_grad()
def eval_mae(model, loader, device):
    """Evaluate MAE (raw ppm values)."""
    was_training = model.training
    model.eval()
    pred, true = evaluate(model, loader, device)

    if was_training:
        model.train()

    return mean_absolute_error(true, pred)


def train_model(config_name, model, train_set, val_set, device,
                epochs=100, bs=32, lr=5e-4, weight_decay=1e-6,
                lambda_reg=0.02, early_stop_patience=20):
    print(f"\n{'=' * 80}\nTraining: {config_name}\n{'=' * 80}")

    # Simple loaders
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, drop_last=False)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_mae = float('inf')
    patience = 0
    os.makedirs(config.DIR_MODELS, exist_ok=True)
    save_path = os.path.join(config.DIR_MODELS, f"{config_name}.pt")

    for epoch in range(1, epochs + 1):
        train_loss, pred_loss, reg_loss = train_epoch(
            model, train_loader, optimizer, device, lambda_reg=lambda_reg
        )

        val_mae = eval_mae(model, val_loader, device)
        scheduler.step(val_mae)

        if (epoch % 20) == 0:
            if reg_loss > 0:
                print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}, reg={reg_loss:.5f}, val={val_mae:.4f} ppm")
            else:
                print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}, val={val_mae:.4f} ppm")

        if val_mae < best_val_mae:
            best_val_mae = val_mae

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'best_val_mae': best_val_mae,
                'epoch': epoch
            }
            torch.save(checkpoint, save_path)
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"  Early stop at epoch {epoch}")
                break

    print(f"Saved best checkpoint -> {save_path} (val MAE: {best_val_mae:.4f} ppm)")
    return save_path, best_val_mae


def main():
    # Global determinism
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.PRINT_DEVICE:
        print(f"Using device: {device}")

    # Dataset & splits
    if not (config.DATASET_PT.exists() and config.SPLIT_NPZ.exists()):
        raise FileNotFoundError(
            f"Missing dataset/splits.\nExpected:\n  {config.DATASET_PT}\n  {config.SPLIT_NPZ}\n"
            "Make sure your working directory is the repo root and that you ran make_splits_C.py."
        )

    data_list = torch.load(config.DATASET_PT, map_location="cpu")
    split = np.load(config.SPLIT_NPZ)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    if config.PRINT_SPLIT_INFO:
        print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    dataset = SimpleDataset(data_list)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    for SEED in config.SEEDS:
        print(f"\n{'=' * 80}\n{'=' * 26} SEED {SEED} {'=' * 26}\n{'=' * 80}")
        # Per-seed determinism
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        # 2D baseline
        model_2d = build_2d().to(device)
        train_model(
            f"C_2D_baseline_seed{SEED}", model_2d, train_set, val_set, device,
            epochs=config.EPOCHS, bs=config.BATCH_SIZE, lr=config.LR, weight_decay=config.WEIGHT_DECAY,
            lambda_reg=config.LAMBDA_REG, early_stop_patience=config.EARLY_PATIENCE)

        # 3D single conformer (from scratch)
        model_3d = build_3d().to(device)
        train_model(
            f"C_3D_single_seed{SEED}", model_3d, train_set, val_set, device,
            epochs=config.EPOCHS, bs=config.BATCH_SIZE, lr=config.LR, weight_decay=config.WEIGHT_DECAY,
            lambda_reg=config.LAMBDA_REG, early_stop_patience=config.EARLY_PATIENCE,
        )

        # 3D single conformer (pretrained init)
        if config.IF_PRETRAIN:
            model_3d_pt = build_3d().to(device)
            ok = maybe_load_pretrained(model_3d_pt, config.PRETRAINED_PATH, device)
            if not ok:
                print("Proceeding without pretrained (weights not loaded).")
            train_model(
                f"C_3D_single_pretrained_seed{SEED}", model_3d_pt,
                train_set, val_set, device,
                epochs=config.EPOCHS, bs=config.BATCH_SIZE, lr=config.LR, weight_decay=config.WEIGHT_DECAY,
                lambda_reg=config.LAMBDA_REG, early_stop_patience=config.EARLY_PATIENCE,
            )

    print("\n" + "=" * 80)
    print("All training runs complete!")
    print(f"Checkpoints saved in: {config.DIR_MODELS}")
    print("=" * 80)


if __name__ == "__main__":
    main()