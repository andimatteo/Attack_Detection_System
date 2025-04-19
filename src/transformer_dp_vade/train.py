# train.py

import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils as utils   # per gradient clipping
import numpy as np
import torch.backends.cudnn as cudnn

from dataset import build_10_packet_blocks, PacketBlockDataset
from model import MiniTransformer, DPVaDE, ClassificationHead

def train_epoch(train_loader, transformer, dpvade, classifier, optimizer, device, writer, global_step,
                alpha_supervised=1.0, max_norm=5.0):
    """
    Esegue un'epoca di training (backprop).
    alpha_supervised: peso della cross entropy supervisionata.
    max_norm: soglia di gradient clipping.
    """
    transformer.train()
    dpvade.train()
    classifier.train()

    running_loss = 0.0

    for i, (blocks, labels) in enumerate(train_loader):
        blocks = blocks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        seq_embed = transformer(blocks)
        total_loss, recon_loss, kl_loss, z = dpvade.compute_loss(seq_embed)

        # CE supervisionata se label >=0
        sup_mask = (labels >= 0)
        if sup_mask.any():
            logits = classifier(seq_embed[sup_mask])  # (num_labeled, n_classes)
            ce_loss = nn.functional.cross_entropy(logits, labels[sup_mask])
        else:
            ce_loss = torch.tensor(0.0, device=device)

        final_loss = total_loss + alpha_supervised * ce_loss
        final_loss.backward()

        # Gradient clipping
        utils.clip_grad_norm_(
            list(transformer.parameters()) + list(dpvade.parameters()) + list(classifier.parameters()),
            max_norm
        )
        optimizer.step()

        running_loss += final_loss.item()

        # Log su TensorBoard ogni tot step
        if (i+1) % 10 == 0:
            writer.add_scalar('Train/total_loss', total_loss.item(), global_step)
            writer.add_scalar('Train/ce_loss', ce_loss.item(), global_step)
            writer.add_scalar('Train/recon_loss', recon_loss.item(), global_step)
            writer.add_scalar('Train/kl_loss', kl_loss.item(), global_step)

        global_step += 1

    avg_loss = running_loss / len(train_loader)
    return avg_loss, global_step

@torch.no_grad()
def evaluate(val_loader, transformer, dpvade, classifier, device, writer=None, epoch=0, alpha_supervised=1.0):
    """
    Valuta su validation set la MSE di ricostruzione + cross entropy per i campioni etichettati.
    """
    transformer.eval()
    dpvade.eval()
    classifier.eval()

    recon_sum = 0.0
    kl_sum = 0.0
    ce_sum = 0.0
    total_samples = 0
    labeled_samples = 0

    for blocks, labels in val_loader:
        blocks = blocks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        seq_embed = transformer(blocks)
        total_loss, recon_loss, kl_loss, z = dpvade.compute_loss(seq_embed)

        bs = blocks.size(0)
        recon_sum += recon_loss.item() * bs
        kl_sum += kl_loss.item() * bs
        total_samples += bs

        sup_mask = (labels >= 0)
        if sup_mask.any():
            logits = classifier(seq_embed[sup_mask])
            y_sup = labels[sup_mask]
            ce_loss = nn.functional.cross_entropy(logits, y_sup, reduction='sum')
            ce_sum += ce_loss.item()
            labeled_samples += sup_mask.sum().item()

    if total_samples > 0:
        avg_recon = recon_sum / total_samples
        avg_kl = kl_sum / total_samples
        avg_vade_loss = avg_recon + avg_kl
    else:
        avg_vade_loss = 0.0

    if labeled_samples > 0:
        avg_ce = ce_sum / labeled_samples
    else:
        avg_ce = 0.0

    avg_combined = avg_vade_loss + alpha_supervised * avg_ce

    if writer:
        writer.add_scalar('Val/VADE_loss', avg_vade_loss, epoch)
        writer.add_scalar('Val/CE_loss', avg_ce, epoch)
        writer.add_scalar('Val/Combined_loss', avg_combined, epoch)

    return avg_vade_loss, avg_ce, avg_combined

def e_m_iteration(data_loader, transformer, dpvade, device, spawn_threshold=0.01, spawn_min_count=50):
    """
    Esegue E-step e M-step su tutto il dataset di training, con possibilità di spawn di nuovi cluster.
    """
    transformer.eval()
    dpvade.eval()

    all_z = []
    for blocks, _ in data_loader:
        blocks = blocks.to(device, non_blocking=True)
        seq_embed = transformer(blocks)
        mu, logvar = dpvade.encode(seq_embed)
        z = dpvade.reparameterize(mu, logvar)
        all_z.append(z)

    if len(all_z) == 0:
        return

    all_z = torch.cat(all_z, dim=0)

    gamma = dpvade.e_step(all_z)
    dpvade.spawn_new_component(all_z, gamma, threshold=spawn_threshold, min_count=spawn_min_count)
    dpvade.m_step(all_z, gamma)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate')
    parser.add_argument('--alpha_supervised', type=float, default=1.0)
    parser.add_argument('--scheduler', type=str, default='none', choices=['none','cosine','onecycle'])
    parser.add_argument('--log_dir', type=str, default='runs/semi_sup')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--csv_path', type=str, default='../../data/dataset1.csv')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Number of DataLoader workers for CPU parallelism')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Use pin_memory=True (utile se device=cuda)')
    parser.add_argument('--spawn_step_freq', type=int, default=1,
                        help='E-step/M-step frequency in epochs (e.g. 1=every epoch)')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # ---- OPZIONE 1: Imposta i thread per PyTorch se sei su CPU
    # (In molti casi, è utile anche su GPU per operazioni su CPU come la GMM)
    # Esempio: usiamo 120 thread su un Intel Xeon con 120 core
    torch.set_num_threads(120)
    torch.set_num_interop_threads(120)

    # OPZIONE 2: Se preferisci controllare via environment:
    # OMP_NUM_THREADS=120 MKL_NUM_THREADS=120 python train.py ...
    # In quel caso commenta set_num_threads

    # Se usi GPU e shape costante, abilita benchmark
    if args.device == 'cuda':
        cudnn.benchmark = True

    device = args.device if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu'
    print(f"[INFO] Using device = {device}")

    print("Loading data from:", args.csv_path)
    blocks, labels, class_map = build_10_packet_blocks(args.csv_path, block_size=10, shuffle_by_class=False)
    print("Blocks shape:", blocks.shape)

    if len(blocks) == 0:
        print("[ERROR] No data found after cleaning. Check CSV or block_size.")
        return

    n_samples = blocks.shape[0]
    n_val = int(0.2 * n_samples)
    n_train = n_samples - n_val

    dataset_full = PacketBlockDataset(blocks, labels)
    train_ds, val_ds = random_split(dataset_full, [n_train, n_val])

    # DataLoader con num_workers elevato (da sperimentare: 16, 32, 64, etc.)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=2,        # Se hai molta RAM, puoi aumentare
        persistent_workers=True   # Mantieni i worker vivi
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=2,
        persistent_workers=True
    )

    n_classes = len([c for c in class_map])  # class_map -> 0..N-1

    d_input = blocks.shape[2]
    # Se hai modificato il MiniTransformer con mean+max pooling, ricordati le dimensioni in uscita!
    # Ad esempio, se d_model=32 e concat mean+max, la dimensione di embedding = 64
    d_model = 32
    transformer = MiniTransformer(
        d_input=d_input,
        d_model=d_model,
        nhead=4,
        num_layers=2,        # se vuoi più strati
        dim_feedforward=64,
        dropout=0.1
    )

    dpvade = DPVaDE(
        input_dim=d_model,  # se hai mean+max -> 2*d_model
        latent_dim=16,
        hidden_dim=64,
        n_components=9,
        alpha_new=1e-3
    )

    classifier = ClassificationHead(
        d_in=d_model,       # se mean+max: 2*d_model
        n_classes=n_classes,
        hidden=64,
        dropout=0.1
    )

    transformer.to(device)
    dpvade.to(device)
    classifier.to(device)

    params = list(transformer.parameters()) + list(dpvade.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs
        )
    else:
        scheduler = None

    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"--- Epoch {epoch}/{args.epochs} ---")
        train_loss, global_step = train_epoch(
            train_loader, transformer, dpvade, classifier,
            optimizer, device, writer, global_step,
            alpha_supervised=args.alpha_supervised,
            max_norm=5.0
        )
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")

        # E-step/M-step ogni X epoche
        if epoch % args.spawn_step_freq == 0:
            e_m_iteration(train_loader, transformer, dpvade, device,
                          spawn_threshold=0.01, spawn_min_count=50)

        vade_loss, ce_loss, combined_loss = evaluate(
            val_loader, transformer, dpvade, classifier, device, writer, epoch,
            alpha_supervised=args.alpha_supervised
        )
        print(f"  Val VADE_loss={vade_loss:.4f}, Val CE={ce_loss:.4f}, combined={combined_loss:.4f}")

        # Scheduler step
        if scheduler is not None:
            if args.scheduler == 'onecycle':
                scheduler.step()
            else:
                scheduler.step()

        # Salvataggio checkpoint se miglioramento
        if combined_loss < best_val_loss:
            best_val_loss = combined_loss
            ckpt_path = os.path.join(args.save_dir, "semi_sup_best.pt")
            print(f"Saving best checkpoint at {ckpt_path} (epoch={epoch})")
            torch.save({
                'epoch': epoch,
                'transformer_state': transformer.state_dict(),
                'dpvade_state': dpvade.state_dict(),
                'classifier_state': classifier.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'n_classes': n_classes,
                'class_map': class_map
            }, ckpt_path)

        # Checkpoint di epoca
        ckpt_epoch = os.path.join(args.save_dir, f"semi_sup_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'transformer_state': transformer.state_dict(),
            'dpvade_state': dpvade.state_dict(),
            'classifier_state': classifier.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, ckpt_epoch)

    writer.close()
    print("Training completed.")

if __name__ == '__main__':
    main()
