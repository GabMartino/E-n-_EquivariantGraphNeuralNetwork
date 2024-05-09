import glob
import os
import pathlib
from pathlib import Path

import torch_geometric as tg
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch_geometric.datasets import QM9
import os

class QM9Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.list_of_files = [os.path.basename(f) for f in glob.glob(self.data_dir + "*.xyz")]
        self.len = len(self.list_of_files)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return

class QM9Dataloader(pl.LightningDataModule):
    def __init__(self, data_dir, splits= [99831, 18000, 13000], batch_size=32):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset = tg.datasets.QM9(root=self.data_dir, force_reload=False)
        print(len(self.dataset), np.sum(splits))
        if (np.sum(splits) != len(self.dataset)) or (type(np.sum(splits)) == np.float32 and  np.sum(splits) > 1.0):
            raise ValueError("Splits sum out of bounds.")

        self.train, self.val, self.test = torch.utils.data.random_split(self.dataset, splits)


    def train_dataloader(self):
        def collate_fn(batch):
            embedding, positions, y = batch.x, batch.pos, batch.y.squeeze()
            Edges = tg.utils.to_dense_adj(batch.edge_index).squeeze()
            return (Edges, positions, embedding), y
        return tg.loader.DataLoader(self.train, batch_size=self.batch_size, num_workers=8, shuffle=True, collate_fn=collate_fn)
    def val_dataloader(self):
        def collate_fn(batch):
            print(batch)
            embedding, positions, y = batch.x, batch.pos, batch.y.squeeze()
            Edges = tg.utils.to_dense_adj(batch.edge_index).squeeze()
            return (Edges, positions, embedding), y
        return tg.loader.DataLoader(self.val, batch_size=self.batch_size, num_workers=8, shuffle=False, collate_fn=collate_fn)
    def test_dataloader(self):
        def collate_fn(batch):
            embedding, positions, y = batch.x, batch.pos, batch.y.squeeze()
            Edges = tg.utils.to_dense_adj(batch.edge_index).squeeze()
            return (Edges, positions, embedding), y
        return tg.loader.DataLoader(self.test, batch_size=self.batch_size, num_workers=8, shuffle=False, collate_fn=collate_fn)







def main():

    dataloader = QM9Dataloader(data_dir="../Datasets/QM9/")
    print(dataloader.train)


if __name__ == "__main__":
    main()