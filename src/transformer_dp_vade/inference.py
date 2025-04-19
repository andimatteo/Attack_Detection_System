# inference.py

import argparse
import torch
from torch.utils.data import DataLoader

import torch.nn.functional as F

from dataset import build_10_packet_blocks, PacketBlockDataset
from model import MiniTransformer, DPVaDE, ClassificationHead

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, required=True)
    parser.add_argument('--csv_test', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--spawn_new_clusters', action='store_true')
    args = parser.parse_args()

    device = args.device if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu'

    ckpt = torch.load(args.model_checkpoint, map_location=device)
    epoch_ckpt = ckpt.get('epoch', 0)
    n_classes = ckpt.get('n_classes', 10)
    class_map = ckpt.get('class_map', {})

    blocks_test, labels_test, class_map_test = build_10_packet_blocks(args.csv_test, block_size=10, shuffle_by_class=False)
    test_ds = PacketBlockDataset(blocks_test, labels_test)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    if blocks_test.shape[0]==0:
        print("No test data available.")
        return

    d_input = blocks_test.shape[2]
    d_model = 32

    transformer = MiniTransformer(d_input=d_input, d_model=d_model)
    dpvade = DPVaDE(input_dim=d_model, latent_dim=16, hidden_dim=64)
    classifier = ClassificationHead(d_model=d_model, n_classes=n_classes)

    transformer.load_state_dict(ckpt['transformer_state'])
    dpvade.load_state_dict(ckpt['dpvade_state'])
    classifier.load_state_dict(ckpt['classifier_state'])

    transformer.to(device).eval()
    dpvade.to(device).eval()
    classifier.to(device).eval()

    print(f"Checkpoint loaded (epoch={epoch_ckpt}). n_classes={n_classes}.")

    recon_sum = 0.0
    ll_sum = 0.0
    count = 0
    all_preds = []
    all_labels = []
    all_z = []

    for blocks, lbls in test_loader:
        blocks = blocks.to(device)
        lbls = lbls.to(device)
        bsize = blocks.size(0)

        seq_embed = transformer(blocks)
        recon, mu, logvar, z = dpvade(seq_embed)

        r_loss = F.mse_loss(recon, seq_embed, reduction='sum').item()
        logpz = dpvade.log_p_z(z)

        recon_sum += r_loss
        ll_sum += logpz.sum().item()
        count += bsize

        # se label >=0 => calcoliamo la predizione
        sup_mask = (lbls >= 0)
        if sup_mask.any():
            logits = classifier(seq_embed[sup_mask])
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(lbls[sup_mask].cpu().tolist())

        all_z.append(z)

    if count>0:
        # Normalizziamo la MSE
        d_out = recon.size(1)
        recon_avg = recon_sum/(count*d_out)
        ll_avg = ll_sum/count
        print(f"[Test Results] Reco MSE={recon_avg:.4f}, log_p(z)={ll_avg:.4f}")

    if len(all_labels)>0:
        correct = sum(p==y for p,y in zip(all_preds, all_labels))
        acc = correct/len(all_labels)
        print(f"Supervised accuracy on labeled portion: {acc*100:.2f}%")

    if args.spawn_new_clusters and len(all_z)>0:
        print("Attempting to spawn new clusters on test data...")
        all_z_cat = torch.cat(all_z, dim=0)
        gamma = dpvade.e_step(all_z_cat)
        dpvade.spawn_new_component(all_z_cat, gamma, threshold=0.01, min_count=50)
        dpvade.m_step(all_z_cat, gamma)
        print(f"Updated n_components = {dpvade.n_components}")

if __name__=='__main__':
    main()
