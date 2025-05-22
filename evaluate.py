"""
    evaluate.py

    This script evaluates the performance of a trained GAN model by computing:
    - Precision and Recall: using the trained Discriminator
    - Frechet Inception Distance (FID): using torchmetrics

    It expects:
    - A directory of generated images (fake samples)
    - A trained Generator and Discriminator checkpoint
    - The MNIST test dataset as the real distribution

    Usage:
        python evaluate.py --dir samples/ --checkpoint checkpoints/
"""

import os
import argparse
import logging
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score
from torchmetrics.image.fid import FrechetInceptionDistance

from models import Generator, Discriminator
from utils import load_model, load_discriminator_model
from loggings import setup_logging

def generate_fake_images(G, total, batch_size=64, latent_dim=100, device='cuda'):
    G.eval()
    fake_images = []

    with torch.no_grad():
        while len(fake_images) < total:
            z = torch.randn(min(batch_size, total - len(fake_images)), latent_dim, device=device)
            fake = G(z).cpu().view(-1, 1, 28, 28)
            fake_images.append(fake)
            torch.cuda.empty_cache()

    return torch.cat(fake_images, dim=0)

def compute_fid(G, real_loader, fake_loader, batch_size, num_samples, device):
    fid = FrechetInceptionDistance(feature=64).to(device)
    fid.reset()

    def preprocess(images):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        images = (images + 1.0) * 127.5
        return images.clamp(0, 255).to(torch.uint8)

    # real images
    count = 0
    for batch, _ in real_loader:
        if count >= num_samples: break
        current_batch = batch[:min(batch_size, num_samples - count)]
        fid.update(preprocess(current_batch).to(device), real=True)
        count += len(current_batch)
        torch.cuda.empty_cache()

    # fake images
    for batch_tuple in fake_loader:
        fake_batch = batch_tuple[0].to(device)
        fid.update(preprocess(fake_batch), real=False)
        torch.cuda.empty_cache()

    return fid.compute().item()

def calculate_precision_recall_f1(D, real_data, fake_data, device):
    real_data = real_data.view(-1, 784).to(device)
    fake_data = fake_data.view(-1, 784).to(device)

    all_data = torch.cat([real_data, fake_data])
    labels = torch.cat([torch.ones(real_data.size(0)), torch.zeros(fake_data.size(0))])

    with torch.no_grad():
        preds = D(all_data)
        preds = (preds > 0.5).float().squeeze()

    precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), zero_division=0)
    recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), zero_division=0)
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), zero_division=0)

    return precision, recall, f1

def evaluate(args):
    """
        Main evaluation routine: computes precision, recall, f1 score, and FID.

        Args:
            args: Parsed command-line arguments
    """
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mnist_dim = 784

    logging.info("Loading models...")
    G = load_model(Generator(g_output_dim=mnist_dim).to(device), args.checkpoint)
    D = load_discriminator_model(Discriminator(mnist_dim).to(device), args.checkpoint)
    G = torch.nn.DataParallel(G).to(device).eval()
    D = torch.nn.DataParallel(D).to(device).eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    logging.info("Loading datasets...")
    real_dataset = datasets.MNIST(root='data/MNIST', train=False, transform=transform, download=True)
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)
    
    real_images = []
    for batch, _ in real_loader:
        real_images.append(batch)
        if sum(x.size(0) for x in real_images) >= args.num_samples:
            break
    real_images = torch.cat(real_images)[:args.num_samples]

    logging.info("Generating fake samples...")
    fake_images = generate_fake_images(G, total=args.num_samples, batch_size=args.batch_size, device=device)
    fake_loader = DataLoader(TensorDataset(fake_images), batch_size=args.batch_size, shuffle=False)
    
    logging.info("Calculating Precision, Recall, and F1...")
    precision, recall, f1_score = calculate_precision_recall_f1(D, real_images, fake_images, device)

    logging.info("Calculating FID...")
    fid = compute_fid(G, real_loader, fake_loader, args.batch_size, num_samples=args.num_samples, device=device)

    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall:    {recall:.4f}")
    logging.info(f"F1 score:  {f1_score:.4f}")
    logging.info(f"FID Score: {fid:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate GAN outputs with precision, recall, and FID.")
    parser.add_argument('--batch_size', type=int, default=64, help="The batch size to use for training.")
    parser.add_argument('--checkpoint', type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument('--num_samples', type=int, default=10000, help="Number of samples.")
    args = parser.parse_args()
    evaluate(args)