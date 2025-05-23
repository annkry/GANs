import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging

from models import Generator, Discriminator
from utils import save_models, load_model, load_discriminator_model
from datasets import NoiseDataset
from loggings import setup_logging

def collaborative_training(train_loader, mnist_dim, args):
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Running in COLLAB mode")

    # load pre-trained Generator and Discriminator models
    G = load_model(Generator(g_output_dim=mnist_dim).to(device), args.checkpoint)
    G = torch.nn.DataParallel(G).to(device)
    D = load_discriminator_model(Discriminator(mnist_dim).to(device), args.checkpoint)
    D = torch.nn.DataParallel(D).to(device)

    criterion = nn.BCELoss()
    noise = NoiseDataset(dim=100)

    # use Adam optimizers
    optim_d = optim.Adam(D.parameters(), lr=args.lr)
    optim_g = optim.Adam(G.parameters(), lr=args.lr)

    rollout_rate = 0.1
    rollout_steps = 50

    # one pass through the dataset
    for batch_idx, (x, _) in tqdm(enumerate(train_loader), desc="Collaborative training"):
        # flatten input images to vectors
        x = x.view(-1, mnist_dim).to(device)
        batch_size = x.size(0)

        # generate fake samples from noise
        noise_batch = noise.next_batch(batch_size).to(device)
        fake_batch = G(noise_batch)

        # prepare refinement buffer
        proba_refine = torch.zeros([batch_size, mnist_dim], device=device)
        proba_steps = torch.randint(0, rollout_steps, (batch_size, 1), device=device)
        proba_steps_one_hot = torch.zeros(batch_size, rollout_steps, device=device).scatter_(1, proba_steps, 1)

        # initialize small perturbations for refinement
        delta_refine = torch.zeros([batch_size, mnist_dim], requires_grad=True, device=device)
        optim_r = optim.Adam([delta_refine], lr=rollout_rate)

        label = torch.ones(batch_size, 1, device=device)  # target: real

        # perform iterative refinement
        for k in range(rollout_steps):
            optim_r.zero_grad()
            output = D(fake_batch.detach() + delta_refine)
            loss_r = criterion(output, label)
            loss_r.backward()
            optim_r.step()

            mask = proba_steps_one_hot[:, k] == 1
            proba_refine[mask] = delta_refine[mask].detach()

        # train Discriminator on real and refined fake samples
        optim_d.zero_grad()
        real_output = D(x)
        loss_d_real = criterion(real_output, label)
        loss_d_real.backward()

        label.fill_(0)  # target: fake
        refined_fake = fake_batch + proba_refine
        fake_output = D(refined_fake.detach())
        loss_d_fake = criterion(fake_output, label)
        loss_d_fake.backward()
        optim_d.step()

        # train Generator to fool the Discriminator using refined samples
        optim_g.zero_grad()
        label.fill_(1)  # target: real
        output = D(refined_fake)
        loss_g = criterion(output, label)
        loss_g.backward()
        optim_g.step()

    # save refined models
    save_models(G, D, args.checkpoint)
    logging.info(f"Collaborative refinement completed, models saved in '{args.checkpoint}'.")
