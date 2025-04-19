# Mini-Transformer con Positional Encoding e DP-VaDE (versione avanzata)

In questa cartella trovi un **sistema di clustering/anomaly detection** basato su:

1. **Mini-Transformer** con *positional encoding* per blocchi di pacchetti (es.: 10 pacchetti).
2. **DP-VaDE** (*Variational Autoencoder* + *Dirichlet Process* / GMM latente) con supporto alla **scoperta proattiva** di nuove classi.
3. **Meccanismi avanzati**:
   - *TensorBoard* per monitorare in tempo reale l'andamento del training.
   - *Checkpoints* per salvare e caricare il modello.
   - *Learning rate scheduler* (OneCycleLR, CosineAnnealingLR).
   - *Metriche di ricostruzione e log-likelihood* (anche su classi sconosciute).

## File Principali

- **`dataset.py`**: Contiene la classe `PacketBlockDataset` e la funzione `build_10_packet_blocks` per generare blocchi di pacchetti dal CSV (`data/dataset.csv`).
- **`model.py`**: 
  - `MiniTransformer` (encoder basato su *self-attention*)  
  - `DPVaDE` (VAE con prior GMM dinamico)  
- **`train.py`**: Script di training completo. Include:
  - Lettura dei parametri da linea di comando (es. epoche, LR, scheduler, percorsi cartelle).
  - Training con backprop.
  - *TensorBoard* logging.
  - Salvataggio checkpoint (periodico o su miglior validazione).
  - Aggiornamento *online* (semi-incrementale) con *E-step/M-step* a ogni epoca (o ogni *N* epoche).
- **`inference.py`**: Script per caricare un checkpoint e processare un CSV di test (ad es. con la decima classe ignota), valutando se emerge un nuovo cluster.

## Requisiti

- Python 3.8+  
- PyTorch >= 1.10  
- NumPy, pandas, scikit-learn, tensorboard

Installa i requisiti con:
```bash
pip install -r ../../requirements.txt
