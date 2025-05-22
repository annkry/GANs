
# Generative Adversarial Networks with Privacy and Collaboration

## Overview

This project presents a well-engineered implementation of **Generative Adversarial Networks (GANs)** trained on MNIST, with advanced features for:

- **Differential Privacy** via [Opacus](https://opacus.ai)
- **Collaborative Refinement** with sample perturbations
- **Latent Space Exploration** and interpolation techniques
- **Evaluation** using **FID**, precision, and recall
- Reproducibility through CLI tools and structured logging

---

## Directory structure

```
GANs/
├── checkpoints/           # Saved models
├── assets/                # Grids for README
├── collab_mode.py         # Training function for collaborative mode
├── train.py               # Training loop (train | diff_privacy | collab modes)
├── generate.py            # Dirichlet or linear interpolations & grid sampling
├── evaluate.py            # F1 score + FID + Precision/Recall
├── models.py              # Generator & Discriminator
├── utils.py               # Training functions & model IO
└── datasets.py            # Custom dataset classes
```

---

## Running the code

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (three modes)

```bash
# standard GAN 
python train.py --mode train --epochs 50

# with opacus differential privacy
python train.py --mode diff_privacy --epochs 50

# perturbation-based collaborative refinement
python train.py --mode train --epochs 100 --checkpoint checkpoint_collab
python train.py --mode collab --checkpoint checkpoint_collab
```

### 3. Generate samples (three modes)

```bash
# interpolated samples via latent Dirichlet
python generate.py --mode interpolate --interpolation_mode dirichlet

# linearly interpolated samples
python generate.py --mode interpolate --interpolation_mode linear

# grid of latent samples
python generate.py --mode grid

# random sampling
python generate.py --mode random --num_samples 10000
```

### 4. Evaluate model

```bash
python evaluate.py --checkpoint checkpoints --num_samples 1000
```

This computes:
- **Precision & Recall** using the trained Discriminator
- **F1 score**
- **FID** using [TorchMetrics](https://torchmetrics.readthedocs.io)

---

## Key Features

- **Differential privacy**
  - Powered by [opacus](https://opacus.ai)
  - Ensures training with formal privacy guarantees

- **Collaborative refinement**
  - Perturbation learning to refine generated samples
  - Simulates semi-federated feedback

- **Latent space analysis**
  - Dirichlet interpolation across 10 latent vectors
  - Grid-based latent sample exploration

- **Evaluation**
  - FID, f1 score, precision, and recall computed

---

<!-- ## Example Results

<!-- This is for classical GAN -->

<!-- ### Generated samples

<p align="center">
  <img src="assets/grid_GAN.png" alt="Sample grid" width="400"/>
</p>

---

### Latent space interpolation

<p align="center">
  <img src="assets/interpolations_grid_GAN.png" alt="Latent interpolation" width="600"/>
</p>

---

| Metric             |      GAN  |   DP GAN |
|--------------------|-----------|----------|
| FID (1000 samples) | ~0.0829   | ~4.9950  |
| Precision          | ~0.0588   | ~0.9851  |
| Recall             | ~0.7560   | ~0.9910  |
| F1 score           | ~0.1091   | ~0.9880  |

<sub>*Evaluated on MNIST with privacy = off. Results may vary with epsilon level.*</sub>

--- --> 