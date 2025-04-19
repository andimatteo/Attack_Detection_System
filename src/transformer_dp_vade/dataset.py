# dataset.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class PacketBlockDataset(Dataset):
    """
    Dataset PyTorch per blocchi di pacchetti. Ogni elemento è (block, label), con:
      - block: shape (block_size, feature_dim)
      - label: classe del blocco, se supervisionato
    """
    def __init__(self, blocks, labels=None):
        self.blocks = torch.tensor(blocks, dtype=torch.float32)
        self.labels = None
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.blocks.shape[0]

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.blocks[idx], self.labels[idx]
        else:
            return self.blocks[idx]

def build_10_packet_blocks(csv_file, block_size=10, shuffle_by_class=False):
    """
    Legge un CSV con colonna 'class' e feature numeriche.
    Ritorna:
       blocks: np.array, shape (num_blocks, block_size, feature_dim)
       labels: np.array, shape (num_blocks,)
       class_to_idx: dict con mapping da 'class' a intero.
    """
    df = pd.read_csv(csv_file)
    df = df.dropna()

    classes = df['class'].unique().tolist()
    class_to_idx = {c: i for i, c in enumerate(classes)}

    feature_cols = [c for c in df.columns if c != 'class']
    X = df[feature_cols].values
    y_str = df['class'].values
    y = np.array([class_to_idx[c] for c in y_str], dtype=np.int64)

    data_frame = df.copy()
    data_frame['class_idx'] = y

    if shuffle_by_class:
        # mescola i dati all'interno di ogni classe
        data_frame = data_frame.groupby('class_idx').apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
    else:
        # ordina per classe
        data_frame = data_frame.sort_values(by='class_idx', ascending=True)

    blocks_list = []
    labels_list = []

    for c_idx in data_frame['class_idx'].unique():
        subset = data_frame[data_frame['class_idx'] == c_idx]
        feats = subset[feature_cols].values
        num_full = len(feats) // block_size
        for i in range(num_full):
            block_data = feats[i*block_size:(i+1)*block_size]
            blocks_list.append(block_data)
            labels_list.append(c_idx)

    blocks_arr = np.stack(blocks_list, axis=0)
    labels_arr = np.array(labels_list, dtype=np.int64)
    return blocks_arr, labels_arr, class_to_idx