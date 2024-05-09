import numpy as np
import pytorch_lightning as pl
import torch
import torch_geometric as tg
import torch.nn as nn

class EnGNNModelLit(pl.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=1e-16):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = model
        self.loss = nn.MSELoss(reduction='mean')
        self.train_loss_epoch_history = []
        self.val_loss_epoch_history = []

    def training_step(self, batch, batch_idx):
        embedding, positions, y = batch.x, batch.pos, batch.y.squeeze()
        edges = tg.utils.to_dense_adj(batch.edge_index).squeeze()
        batch_ptr = batch.ptr
        out = self.model(edges, positions, embedding, batch_ptr)
        loss = self.loss(out, y)
        self.log('train_loss', loss)
        self.train_loss_epoch_history.append(loss)
        return loss


    def on_train_epoch_end(self):

        loss_epoch = np.mean(self.train_loss_epoch_history)
        self.log('train_loss_epoch', loss_epoch)
        self.train_loss_epoch_history.clear()

    def validation_step(self, batch, batch_idx):
        embedding, positions, y = batch.x, batch.pos, batch.y.squeeze()
        edges = tg.utils.to_dense_adj(batch.edge_index).squeeze()
        batch_ptr = batch.ptr
        out = self.model(edges, positions, embedding, batch_ptr)
        loss = self.loss(out, y)
        self.log('val_loss', loss)
        self.val_loss_epoch_history.append(loss)
        return loss

    def on_validation_epoch_end(self):
        loss_epoch = np.mean(self.train_loss_epoch_history)
        self.log('val_loss_epoch', loss_epoch)
        self.val_loss_epoch_history.clear()

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.current_epoch, eta_min=5e-4)
        return {"optimizer": optimizer, "scheduler": scheduler}