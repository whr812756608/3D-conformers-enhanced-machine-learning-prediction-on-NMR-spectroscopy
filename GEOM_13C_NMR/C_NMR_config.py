# C_NMR_config.py - Central configuration for NMR prediction training and testing
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
DIR_MODELS = REPO_ROOT / "GEOM_models"
DIR_DATA = REPO_ROOT / "data"
DIR_PRETRAIN = REPO_ROOT / "GEOM_pretrain" / "pretrained_models"

# Dataset files
DATASET_PT = DIR_DATASETS / "C_dataset_list.pt"
SPLIT_NPZ = DIR_DATASETS / "C_split_indices.npz"
CONF_DIR = DIR_DATA / "conformer_pickles_C_matched"

# Pretraining data
CONF_DIR_PRETRAIN = REPO_ROOT / "GEOM_pretrain" / "conformer_pickles_pretrain"
PRETRAINED_PATH = DIR_PRETRAIN / "3D_pretrained.pt"

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================
# Seeds for reproducibility
SEEDS =[10, 20, 30, 40, 50]

# Pretrained: set to True to use pretrained 3D model initialization
IF_PRETRAIN = True

# ============================================================
# PRETRAINING HYPERPARAMETERS
# ============================================================
PRETRAIN_MODE = 'self_supervised'

# Pretraining epochs and batch size
PRETRAIN_EPOCHS = 35
PRETRAIN_BATCH_SIZE = 32

# Learning rate for pretraining
PRETRAIN_LR = 1e-3

PRETRAIN_WEIGHT_DECAY = 1e-6
PRETRAIN_EARLY_PATIENCE = 30

# Self-supervised learning task weights
SSL_DISTANCE_WEIGHT = 1.0
SSL_DENOISE_WEIGHT = 0.5
SSL_NOISE_LEVEL = 0.10

# Whether to freeze 2D backbone during pretraining
PRETRAIN_FREEZE_2D = True

# ============================================================
# TRAINING HYPERPARAMETERS (Fine-tuning)
# ============================================================
EPOCHS = 100
BATCH_SIZE = 32

# Learning rate
LR = 5e-4
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
    # Input dimensions
    "node_in_dim": 74,
    "edge_in_dim": 9,

    # 2D Backbone architecture
    "node_hidden": 64,
    "mpnn_embed": 256,
    "num_step_message_passing": 5,
    "set2set_steps": 3,

    # 3D Architecture
    "edge_hidden": 128,
    "n_interactions": 4,
    "n_rbf": 32,
    "rbf_cutoff": 8.0,

    # 3D Module flags
    "use_3d": True,
    "use_schnet": True,
    "use_egnn": True,
    "use_comenet": True,
    "comenet_K": 4,
    "comenet_into_2d": False,

    # Readout and regularization
    "readout_hidden": 512,
    "dropout": 0.1
}

# ============================================================
# TESTING SETTINGS
# ============================================================
TOP_N_ATOMS = 30

# ============================================================
# DISPLAY SETTINGS
# ============================================================
PRINT_DEVICE = True
PRINT_SPLIT_INFO = True
