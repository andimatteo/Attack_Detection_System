# inference.py

import argparse
import torch
import os

from dataset import build_10_packet_blocks, PacketBlockDataset
from model import MiniTransformer, DPVaDE
from torch.utils.data import DataLoader

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, required=True, 
                        help='Percorso del file .pt del modello salvato')
    parser.add_argument('--csv_test', type=str, required=True,
                        help='Percorso del CSV di test (ad es. dataset con 10 classi?)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--spawn_new_clusters', action='store_true',
                        help='Se attivato, esegue e_m_iteration e spawna cluster su dati test.')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    # 1) Carichiamo checkpoint
    ckpt = torch.load(args.model_checkpoint, map_location=device)
    # Param fissi del progetto
    d_model = 32
    transformer = MiniTransformer(d_input=12, d_model=d_model, nhead=4, num_layers=1, dim_feedforward=64)
    dpvade = DPVaDE(input_dim=d_model, latent_dim=16, hidden_dim=64, n_components=9, alpha_new=1e-3)

    transformer.load_state_dict(ckpt['transformer_state'])
    dpvade.load_state_dict(ckpt['dpvade_state'])

    transformer.to(device)
    dpvade.to(device)

    print(f"Checkpoint loaded from epoch={ckpt.get('epoch',0)}")

    # 2) Carichiamo CSV test
    blocks, labels, _ = build_10_packet_blocks(args.csv_test, block_size=10, shuffle_by_class=False)
    test_ds = PacketBlockDataset(blocks, labels)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 3) Calcoliamo errore di ricostruzione e log p(z)
    recon_sum = 0.0
    ll_sum = 0.0
    count = 0

    transformer.eval()
    dpvade.eval()

    # memorizziamo z totali per vedere se spawnare nuovi cluster
    all_z = []

    for blocks_batch, _ in test_loader:
        blocks_batch = blocks_batch.to(device)
        recon, mu, logvar, z = dpvade.forward(transformer(blocks_batch))

        r_loss = torch.nn.functional.mse_loss(recon, transformer(blocks_batch), reduction='sum').item()
        logpz = dpvade.log_p_z(z)  # (B,)
        recon_sum += r_loss
        ll_sum += logpz.sum().item()
        count += blocks_batch.size(0)

        all_z.append(z)

    recon_avg = recon_sum/count
    ll_avg = ll_sum/count

    print(f"[Test Results] Reco MSE={recon_avg:.4f} | log p(z)={ll_avg:.4f}")

    # 4) facciamo E-step/M-step su test? (solo se vogliamo scoprire nuove classi in test)
    if args.spawn_new_clusters:
        print("Attempting to spawn new clusters on test data...")
        all_z_cat = torch.cat(all_z, dim=0)
        gamma = dpvade.e_step(all_z_cat)
        dpvade.spawn_new_component(all_z_cat, gamma, threshold=0.01, min_count=50)
        dpvade.m_step(all_z_cat, gamma)
        print(f"Updated n_components = {dpvade.n_components}")

if __name__ == '__main__':
    main()
