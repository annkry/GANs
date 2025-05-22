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

    G = load_model(Generator(g_output_dim=mnist_dim).to(device), args.checkpoint)
    G = torch.nn.DataParallel(G).to(device)
    D = load_discriminator_model(Discriminator(mnist_dim).to(device), args.checkpoint)
    D = torch.nn.DataParallel(D).to(device)

    criterion = nn.BCELoss()
    noise = NoiseDataset(dim=100)

    optim_d = optim.SGD(D.parameters(), lr=args.lr)
    optim_g = optim.SGD(G.parameters(), lr=args.lr)

    rollout_rate = 0.1
    rollout_steps = 50

    for batch_idx, (x, _) in tqdm(enumerate(train_loader), desc="Collaborative training"):
        # resize x from (batch size, 28, 28) to (batch size, 784)
        x = x.view(-1, mnist_dim).to(device)
        batch_size = x.size(0)

        # synthesize noisy samples
        noise_batch = noise.next_batch(batch_size).to(device)
        fake_batch = G(noise_batch)

        # probabilistic refinement
        proba_refine = torch.zeros([batch_size, mnist_dim], device=device)
        proba_steps = torch.randint(0, rollout_steps, (batch_size, 1), device=device)
        # create a one-hot encoded matrix indicating in which iteration each batch item will be assigned some perturbation.
        proba_steps_one_hot = torch.zeros(batch_size, rollout_steps, device=device).scatter_(1, proba_steps, 1)

        # create tensor of small perturbations
        delta_refine = torch.zeros([batch_size, mnist_dim], requires_grad=True, device=device)
        optim_r = optim.Adam([delta_refine], lr=rollout_rate)

        # Define a target label tensor filled with ones, indicating the desired outcome for the discriminator.
        label = torch.ones(batch_size, 1, device=device)

        # Refinement loop to iteratively adjust `delta_refine` over a set number of steps.
        for k in range(rollout_steps):
            optim_r.zero_grad()
            output = D(fake_batch.detach() + delta_refine) # add perturbations to fake samples
            loss_r = criterion(output, label)
            loss_r.backward() # improve the discriminator
            optim_r.step()

            mask = proba_steps_one_hot[:, k] == 1
            # probabilistic assignment: apply the refined perturbation only at the designated step
            proba_refine[mask] = delta_refine[mask].detach()

        # shape D network: maximize log(D(x)) + log(1 - D(R(G(z))))
        optim_d.zero_grad()

        # train with real
        real_output = D(x)
        loss_d_real = criterion(real_output, label)
        loss_d_real.backward()

        # train with refined
        label.fill_(0)
        fake_output = D((fake_batch + proba_refine).detach())
        loss_d_fake = criterion(fake_output, label)
        loss_d_fake.backward()
        optim_d.step()

        # Update G network: maximize log(D(G(z)))
        optim_g.zero_grad()
        label.fill_(1) # fake labels are real for generator cost
        output = D(fake_batch)
        loss_g = criterion(output, label)
        loss_g.backward()
        optim_g.step()

    save_models(G, D, args.checkpoint)
    logging.info(f"Collaborative refinement completed, models saved in '{args.checkpoint}'.")