import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Importa le tue funzioni/ classi
from dataset import build_10_packet_blocks, PacketBlockDataset
from model import MiniTransformer, DPVaDE, ClassificationHead

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help="Percorso del file .pt salvato (e.g. semi_sup_best.pt)")
    parser.add_argument('--csv_test', type=str, required=True,
                        help="Percorso al CSV di test (può avere colonna 'class' oppure no)")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Imposta 'cuda' se hai GPU, altrimenti 'cpu'")
    args = parser.parse_args()

    # Se hai GPU disponibile, usala
    device = args.device if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu'

    # 1) Carichiamo il checkpoint
    print(f"[INFO] Loading checkpoint from {args.model_checkpoint}")
    ckpt = torch.load(args.model_checkpoint, map_location=device)

    # Ricaviamo alcune info (numero classi, ecc.)
    epoch_ckpt = ckpt.get('epoch', 0)
    n_classes = ckpt.get('n_classes', 10)  # fallback se non salvato
    class_map = ckpt.get('class_map', {})  # dict: class_string -> idx (se salvato)

    # 2) Carichiamo il CSV di test
    print(f"[INFO] Loading test data from {args.csv_test}")
    blocks_test, labels_test, class_map_test = build_10_packet_blocks(args.csv_test, block_size=10, shuffle_by_class=False)
    
    if blocks_test.shape[0] == 0:
        print("[WARN] No valid blocks found in CSV (maybe empty or no 10-packet blocks). Exiting.")
        return
    
    # Creiamo Dataset e DataLoader
    test_ds = PacketBlockDataset(blocks_test, labels_test)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    d_input = blocks_test.shape[2]  # numero feature
    d_model = 32  # deve combaciare col training

    # 3) Istanziamo i modelli e carichiamo gli state_dict
    transformer = MiniTransformer(d_input=d_input, d_model=d_model)
    dpvade = DPVaDE(input_dim=d_model, latent_dim=16, hidden_dim=64)
    classifier = ClassificationHead(d_model=d_model, n_classes=n_classes)

    transformer.load_state_dict(ckpt['transformer_state'])
    dpvade.load_state_dict(ckpt['dpvade_state'])
    classifier.load_state_dict(ckpt['classifier_state'])

    transformer.to(device).eval()
    dpvade.to(device).eval()
    classifier.to(device).eval()

    print(f"[INFO] Checkpoint loaded (epoch={epoch_ckpt}), n_classes={n_classes}.")

    # 4) Eseguiamo la classificazione blocco per blocco
    all_preds = []
    all_labels = []
    # Se nel CSV di test c'è la colonna 'class', labels_test conterrà >=0
    # Altrimenti ci saranno -1 (unlabeled)

    for blocks, lbls in test_loader:
        blocks = blocks.to(device)  # shape (B,10,d_input)
        lbls = lbls.to(device)      # shape (B,)

        # Passaggio nel Transformer
        seq_embed = transformer(blocks)
        # Se ci interessa anche la parte di VAE (log p(z), MSE, ecc.), la potremmo calcolare:
        # recon, mu, logvar, z = dpvade(seq_embed)
        
        # Classifier per predizione supervisionata
        logits = classifier(seq_embed)  # (B, n_classes)
        preds = logits.argmax(dim=1)    # classe predetta

        # Registriamo i risultati
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(lbls.cpu().tolist())

    # 5) Se il CSV di test conteneva etichette (lbl >=0), calcoliamo l’accuratezza
    #    altrimenti stampiamo solo le predizioni
    labeled_indices = [i for i,l in enumerate(all_labels) if l >= 0]
    if len(labeled_indices) > 0:
        correct = sum(1 for i in labeled_indices if all_preds[i] == all_labels[i])
        acc = correct / len(labeled_indices)
        print(f"[RESULT] Accuracy su {len(labeled_indices)} blocchi etichettati = {acc*100:.2f}%")

        # Se vuoi, puoi stampare un mini report (pred -> label)
        # Esempio: primi 10
        # for i in labeled_indices[:10]:
        #     print(f"  Block {i}: label={all_labels[i]}, pred={all_preds[i]}")
    # else:
        # print("[INFO] Nessuna etichetta presente nel CSV di test, stampo solo le predizioni (prime 10):")
        # for i, pred in enumerate(all_preds[:10]):
        #     print(f"  Block {i}: pred={pred}")

    # 6) Se vuoi riconvertire la predizione intera in stringhe di classe,
    #    puoi invertire il mapping class_map: idx->class_string
    #    NB: class_map potrebbe avere un ordine diverso.
    idx_to_class = {v: k for k,v in class_map.items()}
    # Esempio: stampare i primi 10 blocchi con la label di stringa:
    # for i in range(min(10, len(all_preds))):
    #     idx_pred = all_preds[i]
    #     class_str = idx_to_class[idx_pred] if idx_pred in idx_to_class else f"unknown_{idx_pred}"
    #     print(f"Block {i} -> predicted class: {class_str}")


if __name__ == '__main__':
    main()
