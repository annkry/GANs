"""
    Handles image generation from a trained Generator model.
    Supports:
    - Random sampling
    - Latent vector interpolation (linear, Dirichlet)
    - Latent space grid visualization
"""

import os
import torch
import logging
import argparse
import numpy as np
from torchvision.utils import save_image, make_grid

from models import Generator
from utils import load_model
from loggings import setup_logging

def generate_images(G, num_samples=10000, latent_dim=100, device='cuda'):
    """Generates random samples using the generator."""
    z = torch.randn(num_samples, latent_dim, device=device)
    return G(z).detach().cpu().view(-1, 1, 28, 28)

def generate_dirichlet_interpolations(G, num_interpolations, interpolation_steps, latent_dim, device):
    """Generates interpolated samples by mixing latent vectors using Dirichlet distribution."""
    images = []

    with torch.no_grad():
        for _ in range(num_interpolations):
            # Sample 10 random latent vectors
            z_vectors = [torch.randn(1, latent_dim, device=device) for _ in range(10)]

            for _ in range(interpolation_steps):
                # Sample Dirichlet weights
                weights = np.random.dirichlet(np.ones(10))
                weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device).view(-1, 1)

                # Convex combination
                z = sum(w * z for w, z in zip(weights_tensor, z_vectors))

                # Generate image
                img = G(z).cpu().view(1, 1, 28, 28)
                images.append(img)

    return torch.cat(images, dim=0)  # shape: [num_interpolations * steps, 1, 28, 28]

def generate_linear_interpolations(G, num_interpolations, interpolation_steps, latent_dim, device):
    """Generates linear interpolations between pairs of latent vectors."""
    G.eval()
    all_samples = []

    with torch.no_grad():
        for _ in range(num_interpolations):
            z1 = torch.randn(1, latent_dim, device=device)
            z2 = torch.randn(1, latent_dim, device=device)

            # Linear interpolation: z = (1 - alpha) * z1 + alpha * z2
            alphas = torch.linspace(0, 1, interpolation_steps, device=device).view(-1, 1)
            interpolated = (1 - alphas) * z1 + alphas * z2  # shape: [interpolation_steps, latent_dim]

            samples = G(interpolated).cpu().view(-1, 1, 28, 28)
            all_samples.append(samples)

    return torch.cat(all_samples, dim=0)  # shape: [num_interpolations * interpolation_steps, 1, 28, 28]

def generate_latent_grid(G, rows=10, cols=10, latent_dim=100, device='cuda'):
    """Creates a grid of generated samples for visual inspection."""
    z = torch.randn(rows * cols, latent_dim, device=device)
    gen_imgs = G(z).detach().cpu().view(-1, 1, 28, 28)
    return make_grid(gen_imgs, nrow=cols, normalize=True)

def main(args):
    """Main function to handle various sample generation modes."""
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    mnist_dim = 784

    try:
        logging.info("Loading generator model...")
        G = Generator(g_output_dim=mnist_dim).to(device)
        G = load_model(G, args.checkpoint)
        G = torch.nn.DataParallel(G).to(device)
        G.eval()
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    logging.info(f"Starting image generation in mode: {args.mode}")

    if args.mode == 'random':
        samples = generate_images(G, args.num_samples, args.latent_dim)
        for idx, img in enumerate(samples):
            save_image(img, os.path.join(args.output_dir, f'{idx:05d}.png'))
    
    elif args.mode == 'interpolate':
        if args.interpolation_mode == 'dirichlet':
            images = generate_dirichlet_interpolations(
            G,
            num_interpolations=args.num_interpolations,
            interpolation_steps=args.steps,
            latent_dim=args.latent_dim,
            device=device
        )
        elif args.interpolation_mode == 'linear':
            images = generate_linear_interpolations(
                G,
                num_interpolations=args.num_interpolations,
                interpolation_steps=args.steps,
                latent_dim=args.latent_dim,
                device=device
            )

        grid = make_grid(images, nrow=args.steps, normalize=True, padding=0)
        output_path = os.path.join(args.output_dir, "interpolations_grid.png")
        save_image(grid, output_path)
        logging.info(f"Saved interpolation grid: {output_path}")

    elif args.mode == 'grid':
        grid = generate_latent_grid(G, latent_dim=args.latent_dim)
        save_image(grid, os.path.join(args.output_dir, 'grid.png'))

    logging.info(f"Image generation completed. Samples saved in '{args.output_dir}'.")

if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser(description='Generate samples using a trained GAN model.')
    parser.add_argument('--mode', choices=['random', 'interpolate', 'grid'], default='interpolate',
                        help="Mode of generation: 'random' for standard sampling, 'interpolate' for Dirichlet interpolation, 'grid' for a montage.")
    parser.add_argument('--batch_size', type=int, default=2048, help="The batch size to use for training.")
    parser.add_argument('--checkpoint', type=str, default="checkpoints", help="Checkpoint directory.")
    parser.add_argument('--output_dir', type=str, default='samples', help="Directory to save generated images.")
    parser.add_argument('--num_samples', type=int, default=10000, help="Total number of images to generate.")
    parser.add_argument('--latent_dim', type=int, default=100, help="Dimensionality of the latent space.")
    parser.add_argument('--steps', type=int, default=20, help="Steps per interpolation.")
    parser.add_argument('--num_vectors', type=int, default=10, help="Number of latent vectors to interpolate.")
    parser.add_argument('--interpolation_mode', choices=['dirichlet', 'linear'], default='linear', help="Interpolation mode.")
    parser.add_argument('--num_interpolations', type=int, default=10, help="How many interpolations to generate.")
    args = parser.parse_args()
    main(args)