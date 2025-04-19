# dataset.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class PacketBlockDataset(Dataset):
    """
    Dataset PyTorch per blocchi di pacchetti.
    Se la label è -1 => campione non supervisionato (semi-supervised).
    """
    def __init__(self, blocks, labels=None):
        self.blocks = torch.tensor(blocks, dtype=torch.float32)
        self.labels = None
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.blocks[idx], -1
        else:
            return self.blocks[idx], self.labels[idx]

def build_10_packet_blocks(csv_file, block_size=10, shuffle_by_class=False):
    """
    Legge un CSV con colonna 'class' (facoltativa); fa pulizia di Inf/NaN; crea blocchi di 10 pacchetti.
    Ritorna (blocks, labels, class_to_idx).
    """
    df = pd.read_csv(csv_file)

    # 1) Sostituisci Inf con NaN e rimuovi righe
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how='any')

    has_class = ('class' in df.columns)
    if has_class:
        classes = df['class'].unique().tolist()
        # Potresti avere righe "strane"? Se sì, filtra
        df = df.dropna(subset=['class'], how='all')
        classes = df['class'].dropna().unique().tolist()

        class_to_idx = {c: i for i, c in enumerate(classes)}
    else:
        class_to_idx = {}

    feature_cols = [c for c in df.columns if c != 'class']
    # (Facoltativo) normalizzazione veloce su ogni colonna
    for col in feature_cols:
        max_val = df[col].abs().max()
        if max_val > 0:  # evitiamo divisioni per 0
            df[col] = df[col]/max_val

    if has_class:
        label_list = []
        for idx_, row in df.iterrows():
            cls_val = row.get('class', None)
            if pd.isna(cls_val):
                label_list.append(-1)
            else:
                label_list.append(class_to_idx.get(cls_val, -1))
        y = np.array(label_list, dtype=np.int64)
    else:
        y = np.full(len(df), -1, dtype=np.int64)

    data_frame = df[feature_cols].copy()
    data_frame['class_idx'] = y

    if shuffle_by_class and has_class:
        labeled_df = data_frame[data_frame['class_idx'] != -1]
        unlabeled_df = data_frame[data_frame['class_idx'] == -1]
        labeled_df = labeled_df.groupby('class_idx')\
                               .apply(lambda x: x.sample(frac=1))\
                               .reset_index(drop=True)
        data_frame = pd.concat([labeled_df, unlabeled_df], ignore_index=True)
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)
    else:
        data_frame = data_frame.sort_values(by='class_idx', ascending=True)

    blocks_list = []
    labels_list = []
    unique_cls = data_frame['class_idx'].unique()
    for c_idx in unique_cls:
        subset = data_frame[data_frame['class_idx'] == c_idx]
        feats = subset[feature_cols].values
        num_full = len(feats)//block_size
        for i in range(num_full):
            block_data = feats[i*block_size:(i+1)*block_size]
            blocks_list.append(block_data)
            labels_list.append(c_idx)

    if len(blocks_list)==0:
        return np.array([]), np.array([]), class_to_idx

    blocks_arr = np.stack(blocks_list, axis=0)
    labels_arr = np.array(labels_list, dtype=np.int64)
    return blocks_arr, labels_arr, class_to_idx
