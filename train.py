import torch
import logging
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine

from models import Generator, Discriminator
from utils import D_train, G_train, save_models, D_train_with_DP, G_train_with_DP, unwrap_state_dict
from loggings import setup_logging
from collab_mode import collaborative_training

def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_dataset = datasets.MNIST(root='data/MNIST', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST', train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main(args):
    setup_logging()
    os.makedirs(args.checkpoint, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_data_loaders(args.batch_size)
    mnist_dim = 784

    G = torch.nn.DataParallel(Generator(g_output_dim=mnist_dim)).to(device)
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).to(device)

    criterion = nn.BCELoss()

    if args.mode == "train":
        logging.info("Running in TRAIN mode")
        
        G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
        D_optimizer = optim.Adam(D.parameters(), lr=args.lr)

        for epoch in trange(1, args.epochs + 1):
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.view(-1, mnist_dim).to(device)
                D_train(x, G, D, D_optimizer, criterion)
                G_train(x, G, D, G_optimizer, criterion)

            if epoch % 10 == 0:
                save_models(G, D, args.checkpoint)
        logging.info(f"Training completed, models saved in '{args.checkpoint}'.")

    elif args.mode == "collab":
        collaborative_training(train_loader, mnist_dim, args)

    elif args.mode == "diff_privacy":
        logging.info("Running in DIFF_PRIVACY mode")

        G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
        D_optimizer = optim.Adam(D.parameters(), lr=args.lr)

        # Attach Privacy Engine to the Discriminator
        privacy_engine = PrivacyEngine()
        D, D_optimizer, private_train_loader = privacy_engine.make_private(
                module=D,
                optimizer=D_optimizer,
                data_loader=train_loader,
                noise_multiplier=1.0,
                max_grad_norm=1.0
        )

        for epoch in trange(1, args.epochs + 1, desc="DP training epochs"):
            G.train()
            D.train()

            for batch_idx, (x, _) in enumerate(private_train_loader):
                x = x.view(-1, mnist_dim).to(device)
                batch_size = x.size(0)

                real_labels = torch.ones(batch_size, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)
                noise = torch.randn(batch_size, 100, device=device)
                fake_data = G(noise)

                # train Discriminator for multiple steps per Generator step
                for _ in range(args.d_steps):
                    D_train_with_DP(x, fake_data, real_labels, fake_labels, D, D_optimizer, criterion)

                # train Generator
                G_train_with_DP(G, D, batch_size, criterion, real_labels, G_optimizer, device)

            if (epoch - 1) % 10 == 0:
                logging.info("Detaching PrivacyEngine before saving...")
                D_clean = Discriminator(mnist_dim).to(device)
                cleaned_state_dict = unwrap_state_dict(D.state_dict())
                D_clean.load_state_dict(cleaned_state_dict)
                save_models(G, D_clean, args.checkpoint)

        logging.info(f"Differentially private training completed, models saved in '{args.checkpoint}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GAN in multiple modes.')
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--mode", type=str, choices=["train", "collab", "diff_privacy"], default="train",
                        help="Training mode: 'train', 'collab', or 'diff_privacy'.")
    parser.add_argument('--checkpoint', type=str, default='checkpoints', help="Directory to save generated images.")
    parser.add_argument("--d_steps", type=int, default=10, help="Number of D steps per G step in diff_privacy mode.")
    args = parser.parse_args()
    main(args)