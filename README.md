
# Generative Adversarial Networks with Privacy and Collaboration

This project presents an  implementation of **Generative Adversarial Networks (GANs)** trained on MNIST, with advanced features for:

- **Differential privacy** via [opacus](https://opacus.ai)
- **Collaborative refinement** with sample perturbations
- **Latent space exploration** and interpolation techniques
- **Evaluation** using FID, F1 score, Precision, and Recall
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
python train.py --mode train --epochs 75 --checkpoint checkpoint_GAN

# with opacus differential privacy
python train.py --mode diff_privacy --epochs 20 --checkpoint checkpoint_DP

# training + perturbation-based collaborative refinement
python train.py --mode collab --epochs 50 --checkpoint checkpoint_collab
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
- **FID** using [torchMetrics](https://torchmetrics.readthedocs.io)

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

### Results comparison

| Model variant       | Generated samples                           | Latent interpolation                           |
|---------------------|---------------------------------------------|-------------------------------------------------|
| **GAN**             | ![GAN Samples](assets/grid_GAN.png)         | ![GAN Interpolation](assets/interpolations_grid_GAN.png)         |
| **GAN + DP**        | ![DP Samples](assets/grid_DP.png)           | ![DP Interpolation](assets/interpolations_grid_DP.png)           |
| **GAN + Collab**    | ![Collab Samples](assets/grid_collab.png)   | ![Collab Interpolation](assets/interpolations_grid_collab.png)   |


---

| Metric             |      GAN  |   GAN with DP | GAN with collab
|--------------------|-----------|---------------|----------------------|
| FID (1000 samples) | ~0.0829   | ~1.1813       | ~0.2264
| Precision          | ~0.0588   | ~0.2719       | ~0.3834
| Recall             | ~0.7560   | ~0.6150       | ~0.9190
| F1 score           | ~0.1091   | ~0.3771       | ~0.5411


---