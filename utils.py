import torch
import os

def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()

    D_output =  D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()

def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()
    y = torch.ones(x.shape[0], 1).cuda()
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()

def D_train_with_DP(real_data, fake_data, real_labels, fake_labels, D, D_optimizer, criterion):
    # train Discriminator
    D_optimizer.zero_grad()
    D.zero_grad()
    real_output = D(real_data)
    fake_output = D(fake_data.detach())

    real_loss = criterion(real_output, real_labels)
    fake_loss = criterion(fake_output, fake_labels)
    D_loss = real_loss + fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.item()

def G_train_with_DP(G, D, batch_size, criterion, real_labels, G_optimizer, device):
    G.zero_grad()
    if isinstance(D, torch.nn.DataParallel):
        D.module.disable_hooks()
    else:
        D.disable_hooks()

    # generate fake data
    noise = torch.randn(batch_size, 100, device=device)
    fake_data = G(noise)

    # generator loss
    fake_output = D(fake_data)
    G_loss = criterion(fake_output, real_labels)
    G_loss.backward()
    G_optimizer.step()
    if isinstance(D, torch.nn.DataParallel):
        D.module.enable_hooks()
    else:
        D.enable_hooks()

def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))

def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'), weights_only=True)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

def load_discriminator_model(D, folder):
    ckpt = torch.load(os.path.join(folder,'D.pth'), weights_only=True)
    D.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return D

def unwrap_state_dict(state_dict):
    """Remove DataParallel and PrivacyEngine prefixes like '_module.module.'"""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("_module.module."):
            new_k = new_k.replace("_module.module.", "")
        elif new_k.startswith("module."):
            new_k = new_k.replace("module.", "")
        elif new_k.startswith("_module."):
            new_k = new_k.replace("_module.", "")
        new_state_dict[new_k] = v
    return new_state_dict