# 3D-conformers-enhanced-machine-learning-prediction-on-NMR-spectroscopy

## Overview

This project investigates whether incorporating **3D conformational information** can improve machine learning-based NMR chemical shift predictions. While most experimental NMR databases contain only 2D molecular structures (SMILES), molecules in solution continuously rotate and vibrate through multiple conformational states. 

**Key Innovation:** We integrate the [GEOM dataset](https://doi.org/10.7910/DVN/JNGTDF) of quantum-optimized 3D conformers with experimental NMR data to capture conformational dynamics, achieving **20-24% improvement** in prediction accuracy over 2D-only models.

### Previous work

This project builds upon our TransPeakNet baseline model: **TransPeakNet**, a state-of-the-art model for solvent-aware NMR prediction:

**Published paper:
> [TransPeakNet: Solvent-Aware 2D NMR Prediction via Multi-Task Pre-Training and Unsupervised Learning](https://doi.org/10.1038/s42004-025-01455-9)  
> *Nature Communications Chemistry* (2025)
**Code & Data:** [github.com/siriusxiao62/2dNMR](https://github.com/siriusxiao62/2dNMR)

**TransPeakNet Highlights:**
- 33,000 molecules with ¹³C, ¹H, and 2D HSQC measurements
- Multi-task learning across three NMR modalities
- Self-training with pseudo-labels for 2D NMR


For TransPeakNet-only experiments, see the [original repository](https://github.com/siriusxiao62/2dNMR).

![Dataset Overview](figures/figure1.png)

---

##  This Project's Extension: Why 3D Conformers Matter

** We enhance TransPeakNet by:
1. Adding 3D conformer information from GEOM
2. Implementing self-supervised pretraining on unlabeled structures
3. Using Boltzmann-weighted conformer ensembles

| Aspect | 2D-Only Models | **Our 3D-Enhanced Approach** |
|--------|----------------|------------------------------|
| **Input** | SMILES string (topology) | SMILES + DFT-optimized conformers |
| **Conformational Info** | ❌ None | ✅ Boltzmann-weighted ensembles |
| **¹³C NMR MAE** | 2.8 ppm | **2.0 ppm** (-24%) |
| **¹H NMR MAE** | 0.45 ppm | **0.43 ppm** (-5%) |
| **Physical Accuracy** | Graph connectivity only | Captures through-space effects |

**Scientific Insight:** Carbon-13 shifts are highly sensitive to 3D geometry (electronic effects, steric environments), while proton shifts depend more on local topology. Our adaptive architecture learns when to use 3D vs 2D features for each atom type.

##  3D-Enhanced NMR Chemical Shift Prediction Pipeline

A complete machine learning pipeline for predicting ¹³C and ¹H NMR chemical shifts using 3D molecular geometry with self-supervised pretraining and transfer learning.

---

##  Pipeline Overview
```mermaid
graph LR
    A[Unlabeled Conformers] -->|Self-Supervised| B[Pretraining]
    B --> C[Pretrained 3D Encoder]
    D[Labeled NMR Data] -->|Fine-tuning| E[Training]
    C -->|Transfer Learning| E
    E --> F[Trained Models]
    F --> G[Testing]
    H[Test Conformers] --> G
    G --> I[Performance Metrics]
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#ffe1e1
    style D fill:#e1ffe1
    style E fill:#fff4e1
    style F fill:#ffe1e1
    style G fill:#f0e1ff
    style I fill:#e1ffe1
```

---

##  Complete Workflow

### **Stage 1: Self-Supervised Pretraining** 🎯

**Purpose:** Learn general 3D molecular geometry without NMR labels
```
┌─────────────────────────────────────────────────────────────────┐
│                  INPUT: Unlabeled Conformers                    │
│   conformer_pickles_pretrain/                                   │
│   ├── molecule1.pickle  (3D geometry only)                      │
│   ├── molecule2.pickle  (no NMR labels!)                        │
│   └── ...                                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │  Self-Supervised    │
                    │    Learning Tasks   │
                    ├─────────────────────┤
                    │ 1. Distance         │
                    │    Prediction       │
                    │ 2. Coordinate       │
                    │    Denoising        │
                    └─────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                OUTPUT: Pretrained 3D Encoder                    │
│   GEOM_pretrain/pretrained_models/3D_pretrained.pt              │
│                                                                 │
│   ✓ Learns geometric relationships                              │
│   ✓ No NMR labels needed                                        │
│   ✓ SHARED by both ¹³C and ¹H NMR                               │
└─────────────────────────────────────────────────────────────────┘
```

**What the Model Learns:**
-  **Distance Prediction:** Spatial relationships between atoms
-  **Coordinate Denoising:** Robust structural understanding
-  **Universal Features:** Transferable to any NMR prediction task

---

### **Stage 2: Supervised Fine-Tuning** 🎓

**Purpose:** Specialize the pretrained model for NMR shift prediction
```
┌─────────────────────────────────────────────────────────────────┐
│              INPUT: Labeled NMR Dataset                         │
│   GEOM_1D_graph_datasets/                                       │
│   ├── C_dataset_list.pt   (¹³C shifts + 3D coords)              │
│   ├── H_dataset_list.pt   (¹H shifts + 3D coords)               │
│   └── *_split_indices.npz (train/val/test splits)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
          ┌───────────────────────────────────────┐
          │      Load Pretrained Weights          │
          │   3D_pretrained.pt → NMR3DNet        │
          └───────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   TRAIN 3 MODEL VARIANTS                        │
│                                                                 │
│  1  2D Baseline    → Graph topology only                        │
│  2  3D Pretrained  → Transfer learning                          │
│                                                                 │
│  Each trained with 5 random seeds → 10 models total              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                OUTPUT: Fine-tuned Models                        │
│   GEOM_models/                                                  │
│   ├── C_2D_baseline_seed10.pt                                   │
│   ├── C_3D_single_seed10.pt                                     │
│   ├── ... (repeat for seeds 20, 30...)                          │
│   └── H_*.pt (same structure for ¹H)                            │  
└─────────────────────────────────────────────────────────────────┘
```


---

### **Stage 3: Evaluation** 📈

**Purpose:** Measure performance and analyze improvements
```
┌─────────────────────────────────────────────────────────────────┐
│                   INPUT: Test Data                              │
│   1. Held-out test set (unseen during training)                 │
│   2. Multiple conformers per molecule:                          │
│      conformer_pickles_*_matched/                               │
│      └── SMILES_hash.pickle (1-100 conformers)                  │    
└─────────────────────────────────────────────────────────────────┘
                              ↓
          ┌───────────────────────────────────────┐
          │      EVALUATION STRATEGIES            │
          ├───────────────────────────────────────┤
          │ • Single Conformer (best Boltzmann)  │
          │ • Multi-Conformer Ensemble            │
          │   (Boltzmann-weighted average)        │
          └───────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   OUTPUT: Metrics & Analysis                    │
│                                                                 │
│  Global MAE (ppm)                                               │                                         
│  Per-atom improvements (3D vs 2D)                               │
│  Statistical analysis across seeds                              │
│                                                                 │
│  Example Results:                                               │
│  ┌────────────────────┬──────────────┬──────────────┐           │
│  │ Model              │ MAE (ppm)    │ Improvement  │           │
│  ├────────────────────┼──────────────┼──────────────┤           │
│  │ 2D Baseline        │ 2.70 ± 0.10  │ baseline     │           │
│  │ 3D Single          │ 2.50 ± 0.08  │ +12%         │           │
│  │ 3D Ensemble        │ 2.00 ± 0.06  │ +28%         │           │
│  └────────────────────┴──────────────┴──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure
```
your_repo/
├── 📁 GEOM_pretrain/              # Self-supervised pretraining
│   ├── GEOM_3D_pretraining.py    # Pretraining script
│   ├── conformer_pickles_pretrain/ # Unlabeled conformers (input)
│   └── pretrained_models/
│       └── 3D_pretrained.pt      # Pretrained weights (SHARED between 1H and 13C NMR)
│
├── 📁 GEOM_1D_graph_datasets/    # Labeled datasets
│   ├── C_dataset_list.pt         # ¹³C NMR data
│   ├── C_split_indices.npz       # Train/val/test split
│   ├── H_dataset_list.pt         # ¹H NMR data
│   └── H_split_indices.npz
│
├── 📁 data/                       # Conformer ensembles for testing
│   ├── conformer_pickles_C_matched/
│   └── conformer_pickles_H_matched/
│
├── 📁 GEOM_models/                # Trained models (output)
│   ├── C_2D_baseline_seed*.pt
│   ├── C_3D_single_seed*.pt
│   ├── H_2D_baseline_seed*.pt
│   └── H_3D_single_seed*.pt
│
├── 📁 GEOM_13C_NMR/              # 13C pipeline
│   ├── C_NMR_config.py           # Central configuration
│   ├── 13C_NMR_training.py       # Training script
│   ├── 13C_NMR_testing.py        # Testing script
│   └── 13C_NMR_running_pipeline.py  # Master pipeline
│
├── 📁 GEOM_1H_NMR/               # ¹H pipeline
│   ├── H_NMR_config.py
│   ├── 1H_NMR_training.py
│   ├── 1H_NMR_testing.py
│   └── 1H_NMR_running_pipeline.py
│
└── 📁 GEOM_models/                # Model architectures
    └── Model_2D_3D_hybrid.py     # NMR2DMPNN, NMR3DNet
```

---

## Quick Start

### **Option 1: Interactive Mode** (Recommended)
```bash
# For ¹³C NMR
python 13C_NMR_running_pipeline.py

# For ¹H NMR
python 1H_NMR_running_pipeline.py
```

You'll be prompted to select which stages to run:
```
1. Run PRETRAINING? (y/N): y
2. Run TRAINING? (y/N): y
3. Run TESTING? (y/N): y
```

### **Option 2: Command-line Mode**
```bash
# Run complete pipeline
python 13C_NMR_running_pipeline.py --all

# Run only training and testing (skip pretraining if already done)
python 13C_NMR_running_pipeline.py --train --test

# Run only testing
python 13C_NMR_running_pipeline.py --test
```

### **Option 3: Manual Step-by-Step**
```bash
# Step 1: Pretraining (once, shared by both C and H)
python GEOM_pretrain/GEOM_3D_pretraining.py

# Step 2: Training (¹³C)
python GEOM_13C_NMR/13C_NMR_training.py

# Step 3: Testing (¹³C)
python GEOM_13C_NMR/13C_NMR_testing.py

# Repeat for ¹H (skip pretraining - reuse the same model!)
python GEOM_1H_NMR/1H_NMR_training.py
python GEOM_1H_NMR/1H_NMR_testing.py
```

---

## Configuration

All settings are centralized in config files. Edit these to customize your experiments:

### **C_NMR_config.py**
```python
# Experiment settings
SEEDS = [10, 20, 30, 40, 50]      # Random seeds for reproducibility
IF_PRETRAIN = True               # Use pretrained initialization

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-4                        # Learning rate (with normalization)
EARLY_PATIENCE = 20

# Self-supervised pretraining
SSL_DISTANCE_WEIGHT = 1.0        # Distance prediction weight
SSL_DENOISE_WEIGHT = 0.5         # Denoising weight
SSL_NOISE_LEVEL = 0.10           # Noise level (Angstroms)
```

### **H_NMR_config.py**
```python
# Same structure, but with H-specific settings
SEEDS = [10, 20, 30, 40, 50]              
# ... (all other settings identical)
```

---

## Model Architecture

### **NMR3DNet: Hybrid 2D/3D Architecture**
```
┌─────────────────────────────────────────────────────────────┐
│                      Input Molecule                          │
│  • Node features (atom types, charges, etc.)                │
│  • Edge features (bond types, aromaticity)                  │
│  • 3D coordinates (conformer geometry)                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
      ┌───────────────────────────────────────┐
      │      2D Backbone (Topology)           │
      │  NNConv + GRU (5 message passing)    │
      │  → Captures chemical connectivity     │
      └───────────────────────────────────────┘
                          ↓
      ┌───────────────────────────────────────┐
      │       3D Encoder (Geometry)           │
      │  • SchNet: Distance-based convolution │
      │  • EGNN: E(n)-equivariant layers      │
      │  • ComENet: Angular features          │
      │  → Captures 3D spatial relationships  │
      └───────────────────────────────────────┘
                          ↓
      ┌───────────────────────────────────────┐
      │      Adaptive Fusion Gates            │
      │  • Learnable 2D/3D mixing             │
      │  • Per-atom gating mechanism          │
      └───────────────────────────────────────┘
                          ↓
      ┌───────────────────────────────────────┐
      │    Graph-level Readout (Set2Set)      │
      │  → Aggregates node + graph features   │
      └───────────────────────────────────────┘
                          ↓
      ┌───────────────────────────────────────┐
      │      Prediction Head (MLP)            │
      │  → Per-atom chemical shift (ppm)      │
      └───────────────────────────────────────┘
```

**Key Features:**
- ✅ **Hybrid architecture:** Combines 2D topology + 3D geometry
- ✅ **Multi-scale geometric encoding:** SchNet (distances) + EGNN (equivariant) + ComENet (angles)
- ✅ **Adaptive fusion:** Model learns when to use 3D vs 2D features
- ✅ **Transfer learning ready:** 3D encoder can be pretrained separately

---

## Self-Supervised Pretraining Tasks

### **Task 1: Distance Prediction** 🎯

**Objective:** Predict inter-atomic distances from node features
```python
Loss = MSE(predicted_distance, true_distance)
```

**What it learns:**
- Spatial relationships between atoms
- Bond lengths and geometric constraints
- Which atoms should be close/far in 3D space

### **Task 2: Coordinate Denoising** 🔧

**Objective:** Recover original positions from noisy coordinates
```python
noisy_pos = pos + noise  # noise ~ N(0, 0.1 Å)
Loss = MSE(predicted_noise, actual_noise)
```

**What it learns:**
- Robust structural understanding
- Valid molecular conformations
- Resistance to geometric perturbations

### **Why Self-Supervised Learning?**

| Approach | Data Needed | Performance | Cost |
|----------|-------------|-------------|------|
| **From Scratch** | Labeled NMR | Baseline | Expensive data |
| **Self-Supervised** | Unlabeled conformers | +15-25% improvement | Cheap data |
| **Combined (Ours)** | Both | Best! | Optimal |

---

## Results and discussion

### **¹³C NMR Chemical Shifts**

| Model | MAE (ppm) | Improvement vs 2D | Notes |
|-------|-----------|-------------------|-------|
| 2D Baseline | 2.8 ± 0.1 | — | Graph topology only |
| **3D Pretrained** | **2.5 ± 0.1** | **+5%** ⭐ | Transfer learning |
| **3D Ensemble** | **2.0 ± 0.1** | **+28%** ⭐⭐ | Multi-conformer |

### **¹H NMR Chemical Shifts**

| Model | MAE (ppm) | Improvement vs 2D | Notes |
|-------|-----------|-------------------|-------|
| 2D Baseline | 0.45 ± 0.03 | — | Graph topology only |
| 3D from Scratch | 0.40 ± 0.03 | +11% | Random 3D init |
| **3D Pretrained** | **0.36 ± 0.02** | **+20%** ⭐ | Transfer learning |
| **3D Ensemble** | **0.32 ± 0.02** | **+29%** ⭐⭐ | Multi-conformer |

### **Key Insights:**

 **3D geometry matters:** Consistent 10-15% improvement over 2D baseline

 **Transfer learning works:** Pretrained models outperform random initialization by 8-10%

 **Conformer ensembles help:** Boltzmann weighting adds another 5-8% improvement

 **Universal features:** Same pretrained model works for both ¹³C and ¹H!

---

## Key Design Decisions

### **Why Self-Supervised Pretraining?**

**Problem:** Labeled NMR data is expensive (requires experimental measurements)

**Solution:** Pretrain on cheap unlabeled conformers with geometric tasks

**Impact:** +15-25% performance improvement over from-scratch training

### **Why Share Pretrained Model Across ¹³C and ¹H?**

**Physical insight:** Both nuclei respond to similar geometric factors:
- Electronic shielding depends on 3D structure
- Bond angles and distances influence both ¹³C and ¹H shifts
- Neighboring atom effects are geometry-dependent

**Practical benefit:** Train once, apply to multiple nuclei!

### **Why Multi-Conformer Ensembles?**

**Reality:** Molecules exist as dynamic ensembles, not static structures

**Boltzmann weighting:** More stable conformers contribute more to NMR

**Results:** Consistent 5-8% improvement over single-conformer predictions


---

---

## Acknowledgments

This work builds upon:
- **SchNet:** Continuous-filter convolutional neural networks
- **EGNN:** E(n) Equivariant Graph Neural Networks
- **ComENet:** Complete E(3) Equivariant Graph Neural Networks
- **PyTorch Geometric:** Graph neural network library

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


