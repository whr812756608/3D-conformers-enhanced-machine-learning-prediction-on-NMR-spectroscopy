# train_C.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error

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


def train_epoch(model, loader, optimizer, device, y_mean, y_std, lambda_reg=0.02):
    """Training with optional regularization for adaptive models"""
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_reg_loss = 0.0

    for data in loader:
        data = data.to(device)
        pred = model(data)
        target = data.y[data.mask]

        pred_norm = (pred - y_mean) / (y_std + 1e-8)
        target_norm = (target - y_mean) / (y_std + 1e-8)

        pred_loss = torch.nn.functional.l1_loss(pred_norm, target_norm)
        loss = pred_loss

        # Regularization for adaptive models (kept generic; no-op for your fixed models)
        if hasattr(model, 'selector') and getattr(model, 'use_adaptive', False):
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

    n = max(1, len(loader))
    avg_reg = (total_reg_loss / n) if total_reg_loss > 0 else 0.0
    return total_loss / n, total_pred_loss / n, avg_reg


def train_model(config_name, model, dataset, device, train_set, val_set, test_set,
                epochs=100, is_adaptive=False, lambda_reg=0.02):
    """Train model with fixed splits and save best checkpoint"""
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
    y_std = torch.tensor(train_y.std() + 1e-8, dtype=torch.float32, device=device)

    print(f"Training labels: mean={float(y_mean):.2f}, std={float(y_std):.2f}, n={len(train_y)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    lr = 5e-4 if is_adaptive else 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 10, min_lr=1e-6)

    best_val_mae = float('inf')
    patience = 0
    save_path = f"./GEOM_model/{config_name}.pt"

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
                    f"  Epoch {epoch:3d}: loss={train_loss:.4f} (pred={pred_loss:.4f}, reg={reg_loss:.5f}), val={val_mae:.4f}"
                )
            else:
                print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}, val={val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), save_path)
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                print(f"  Early stop at epoch {epoch}")
                break

    # Final test on best checkpoint
    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_pred, test_true = evaluate(model, test_loader, device)
    test_mae = mean_absolute_error(test_true, test_pred)

    print(f"Results: val={best_val_mae:.4f}, test={test_mae:.4f} ppm")
    return model, best_val_mae, test_mae


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("./GEOM_model", exist_ok=True)

    # Optional: pretrained 3D encoder (set to True and update path if used)
    PRETRAINED_PATH = "./GEOM_pretrain/pretrained_models/3D_pretrained.pt"
    USE_PRETRAINED = False

    # Build dataset
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

    # Fixed split (save indices so evaluation can reproduce the same split)
    n = len(dataset)
    torch.manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [int(0.8 * n), int(0.1 * n), n - int(0.8 * n) - int(0.1 * n)]
    )

    # Save split indices for evaluation reproducibility
    split_path = "./GEOM_model/C_split_indices.npz"
    np.savez(
        split_path,
        train_idx=np.array(train_set.indices, dtype=np.int64),
        val_idx=np.array(val_set.indices, dtype=np.int64),
        test_idx=np.array(test_set.indices, dtype=np.int64),
    )
    print(f"Saved split indices -> {split_path}")

    # ========== Train 2D baseline ==========
    print(f"\n{'=' * 80}\nTRAINING 2D BASELINE\n{'=' * 80}")

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
    print(f"\n{'=' * 80}\nTRAINING 3D FIXED MODEL\n{'=' * 80}")

    model_3d = NMR3DNet(
        node_hidden=64, edge_hidden=128, n_interactions=4,
        use_3d=True, use_schnet=True, use_egnn=True, use_comenet=True, comenet_K=4
    ).to(device)

    if USE_PRETRAINED and os.path.exists(PRETRAINED_PATH):
        ckpt = torch.load(PRETRAINED_PATH, map_location=device, weights_only=False)
        try:
            model_3d.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(f"Loaded 3D pretrained encoder from {PRETRAINED_PATH}")
        except Exception as e:
            print(f"⚠ Failed to load pretrained weights: {e}")

    model_3d, val_3d, test_3d = train_model(
        "C_3D_fixed", model_3d, dataset, device, train_set, val_set, test_set,
        is_adaptive=False
    )

    print("\nDone training. Saved:")
    print("  ./GEOM_model/C_2D_baseline.pt")
    print("  ./GEOM_model/C_3D_fixed.pt")


if __name__ == "__main__":
    main()
