# 13C_NMR_config.py - Central configuration for NMR prediction training and testing
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
DATASET_PT   = DIR_DATASETS / "C_dataset_list.pt"
SPLIT_NPZ    = DIR_DATASETS / "C_split_indices.npz"
CONF_DIR     = DIR_DATA / "conformer_pickles_C_matched"

# Pretrained model
PRETRAINED_PATH = DIR_PRETRAIN / "3D_pretrained.pt"

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================
# Seeds for reproducibility (used in both training and testing)
SEEDS = [0, 42, 66]

# Normalization: set to True to normalize labels (recommended)
USE_NORMALIZATION = True

# Pretrained: set to True to use pretrained 3D model initialization
IF_PRETRAIN = True

# ============================================================
# PRETRAINING HYPERPARAMETERS
# ============================================================
# Pretraining usually needs more epochs since training from scratch
PRETRAIN_EPOCHS = 60
PRETRAIN_BATCH_SIZE = 32

# Learning rate for pretraining (depends on normalization)
if USE_NORMALIZATION:
    PRETRAIN_LR = 1e-4  # Smaller LR for normalized labels
else:
    PRETRAIN_LR = 5e-4  # Standard LR for raw labels

PRETRAIN_WEIGHT_DECAY = 1e-6
PRETRAIN_EARLY_PATIENCE = 30  # More patience for pretraining

# ============================================================
# TRAINING HYPERPARAMETERS (Fine-tuning)
# ============================================================
EPOCHS = 100
BATCH_SIZE = 32

# Learning rate depends on normalization
# Smaller LR needed for normalized labels (gradients are scaled by std)
if USE_NORMALIZATION:
    LR = 1e-4  # Smaller LR for normalized labels
else:
    LR = 5e-4  # Standard LR for raw labels

WEIGHT_DECAY = 1e-6
EARLY_PATIENCE = 20

# Regularization
LAMBDA_REG = 0.02
MAX_GRAD_NORM = 5.0

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
# 2D Model settings
MODEL_2D_CONFIG = {
    "node_feats": 64,
    "embed_feats": 256,
    "num_step_message_passing": 5,
    "num_step_set2set": 3,
    "hidden_feats": 512,
    "prob_dropout": 0.1
}

# 3D Model settings
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