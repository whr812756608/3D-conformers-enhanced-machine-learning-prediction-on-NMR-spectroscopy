# H_NMR_config.py - Central configuration for ¹H NMR prediction training and testing
from pathlib import Path
import sys

# ============================================================
# REPO STRUCTURE
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Directories
DIR_DATASETS = REPO_ROOT / "GEOM_1D_graph_datasets"
DIR_MODELS   = REPO_ROOT / "GEOM_models"
DIR_DATA     = REPO_ROOT / "data"
DIR_PRETRAIN = REPO_ROOT / "GEOM_pretrain" / "pretrained_models"

# Dataset files
DATASET_PT   = DIR_DATASETS / "H_dataset_list.pt"
SPLIT_NPZ    = DIR_DATASETS / "H_split_indices.npz"
CONF_DIR     = DIR_DATA / "conformer_pickles_H_matched"


PRETRAINED_PATH = DIR_PRETRAIN / "3D_pretrained.pt"
# ============================================================
# EXPERIMENT SETTINGS
# ============================================================
# Seeds for reproducibility (used in both training and testing)
SEEDS = [10, 20, 30, 40, 50]

# Pretrained: set to True to use pretrained 3D model initialization
# NOTE: The same pretrained weights work for BOTH C and H NMR!
IF_PRETRAIN = True

# ============================================================
# PRETRAINING HYPERPARAMETERS
# ============================================================
# H NMR uses the same pretrained model as C NMR
PRETRAIN_EPOCHS = 35
PRETRAIN_BATCH_SIZE = 32
PRETRAIN_LR = 1e-3
PRETRAIN_WEIGHT_DECAY = 1e-6
PRETRAIN_EARLY_PATIENCE = 30

# Self-supervised learning task weights (same as C NMR)
SSL_DISTANCE_WEIGHT = 1.0
SSL_DENOISE_WEIGHT = 0.5
SSL_NOISE_LEVEL = 0.10

# ============================================================
# TRAINING HYPERPARAMETERS (Fine-tuning)
# ============================================================
EPOCHS = 100
BATCH_SIZE = 32

LR = 5e-4  # Standard LR for raw labels

WEIGHT_DECAY = 1e-6
EARLY_PATIENCE = 20

# Regularization
LAMBDA_REG = 0.02
MAX_GRAD_NORM = 5.0

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
# 2D Model settings (same as C NMR)
MODEL_2D_CONFIG = {
    "node_feats": 64,
    "embed_feats": 256,
    "num_step_message_passing": 5,
    "num_step_set2set": 3,
    "hidden_feats": 512,
    "prob_dropout": 0.1
}

# 3D Model settings (same as C NMR)
MODEL_3D_CONFIG = {
    "node_hidden": 64,
    "edge_hidden": 128,
    "n_interactions": 4,
    "use_3d": True,
    "use_schnet": True,
    "use_egnn": True,
    "use_comenet": True,
    "comenet_K": 4
}

# ============================================================
# TESTING SETTINGS
# ============================================================
TOP_N_ATOMS = 30  # Number of top improved atoms to display

# ============================================================
# DISPLAY SETTINGS
# ============================================================
PRINT_DEVICE = True
PRINT_SPLIT_INFO = True
