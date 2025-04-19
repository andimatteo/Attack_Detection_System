# train.py

import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from dataset import build_10_packet_blocks, PacketBlockDataset
from model import MiniTransformer, DPVaDE

def train_epoch(train_loader, transformer, dpvade, optimizer, device, writer, global_step):
    """
    Esegue un'epoca di training (backprop).
    Ritorna la loss media su train.
    """
    transformer.train()
    dpvade.train()
    running_loss = 0.0

    for i, (blocks, _) in enumerate(train_loader):
        blocks = blocks.to(device)
        optimizer.zero_grad()
        seq_embed = transformer(blocks)  # (B, d_model)
        total_loss, recon_loss, kl_loss, z = dpvade.compute_loss(seq_embed)
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        # Log su TensorBoard ogni tot step
        if (i+1) % 10 == 0:
            writer.add_scalar('Train/total_loss', total_loss.item(), global_step)
            writer.add_scalar('Train/recon_loss', recon_loss.item(), global_step)
            writer.add_scalar('Train/kl_loss', kl_loss.item(), global_step)
        global_step += 1

    avg_loss = running_loss/len(train_loader)
    return avg_loss, global_step

@torch.no_grad()
def evaluate(val_loader, transformer, dpvade, device, writer=None, epoch=0):
    """
    Valuta su validation set la MSE di ricostruzione + log-likelihood
    Ritorna (avg_recon, avg_ll).
    """
    transformer.eval()
    dpvade.eval()

    recon_sum = 0.0
    ll_sum = 0.0
    count = 0

    for blocks, _ in val_loader:
        blocks = blocks.to(device)
        seq_embed = transformer(blocks)
        recon, mu, logvar, z = dpvade.forward(seq_embed)
        # recon_loss
        r_loss = nn.functional.mse_loss(recon, seq_embed, reduction='sum').item()
        # log-likelihood p(z)
        logpz = dpvade.log_p_z(z)  # shape (B,)
        ll_sum += logpz.sum().item()
        recon_sum += r_loss
        count += blocks.size(0)

    avg_recon = recon_sum/(count)
    avg_ll = ll_sum/(count)

    if writer:
        writer.add_scalar('Val/reconstruction', avg_recon, epoch)
        writer.add_scalar('Val/log_pz', avg_ll, epoch)

    return avg_recon, avg_ll

def e_m_iteration(data_loader, transformer, dpvade, device, spawn_threshold=0.01, spawn_min_count=50):
    """
    E-step / M-step su tutto il dataset di training, con possibile spawn di nuove comp.
    """
    transformer.eval()
    dpvade.eval()
    all_z = []
    for blocks, _ in data_loader:
        blocks = blocks.to(device)
        mu, logvar = dpvade.encode(transformer(blocks))
        z = dpvade.reparameterize(mu, logvar)
        all_z.append(z)
    all_z = torch.cat(all_z, dim=0)  # shape (N,latent_dim)

    gamma = dpvade.e_step(all_z)
    dpvade.spawn_new_component(all_z, gamma, threshold=spawn_threshold, min_count=spawn_min_count)
    dpvade.m_step(all_z, gamma)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='none', 
                        choices=['none','cosine','onecycle'])
    parser.add_argument('--log_dir', type=str, default='runs/transformer_dp_vade')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--csv_path', type=str, default='../../data/dataset1.csv')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("Loading data from:", args.csv_path)
    blocks, labels, class_map = build_10_packet_blocks(args.csv_path, block_size=10, shuffle_by_class=False)
    print("Blocks shape:", blocks.shape)

    # Train/val split
    n_samples = blocks.shape[0]
    n_val = int(0.2*n_samples)
    n_train = n_samples - n_val

    dataset_full = PacketBlockDataset(blocks, labels)
    train_ds, val_ds = torch.utils.data.random_split(dataset_full, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # Inizializza modelli
    d_input = blocks.shape[2]  # 12 feature?
    transformer = MiniTransformer(d_input=d_input, d_model=32, nhead=4, num_layers=1, dim_feedforward=64)
    dpvade = DPVaDE(input_dim=32, latent_dim=16, hidden_dim=64, n_components=9, alpha_new=1e-3)

    transformer.to(device)
    dpvade.to(device)

    # Ottimizzatore
    params = list(transformer.parameters()) + list(dpvade.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
    else:
        scheduler = None

    global_step = 0
    best_val_recon = float('inf')
    for epoch in range(1, args.epochs+1):
        # 1) Train
        train_loss, global_step = train_epoch(train_loader, transformer, dpvade, optimizer, device, writer, global_step)
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")

        # 2) E-step / M-step + spawn
        e_m_iteration(train_loader, transformer, dpvade, device, spawn_threshold=0.01, spawn_min_count=50)

        # 3) Validation
        val_recon, val_ll = evaluate(val_loader, transformer, dpvade, device, writer, epoch)
        print(f"    Val Recon: {val_recon:.4f}, Val LogPz: {val_ll:.4f}")

        writer.add_scalar('Val/recon_loss', val_recon, epoch)
        writer.add_scalar('Val/log_pz',     val_ll, epoch)

        # 4) Scheduler step
        if scheduler is not None:
            if args.scheduler == 'onecycle':
                scheduler.step()
            else:
                # Cosine, etc.
                scheduler.step()

        # Salvataggio checkpoint se la ricostruzione è migliorata
        if val_recon < best_val_recon:
            best_val_recon = val_recon
            ckpt_path = os.path.join(args.save_dir, f"dpvade_best.pt")
            print(f"Saving best checkpoint at {ckpt_path} (epoch={epoch})")
            torch.save({
                'epoch': epoch,
                'transformer_state': transformer.state_dict(),
                'dpvade_state': dpvade.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, ckpt_path)

        # Salvataggio checkpoint periodico
        ckpt_epoch = os.path.join(args.save_dir, f"dpvade_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'transformer_state': transformer.state_dict(),
            'dpvade_state': dpvade.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, ckpt_epoch)

    writer.close()
    print("Training completed.")

if __name__ == '__main__':
    main()
