import hydra
import torch
import torch.nn as nn
import torch_geometric as tg

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from En_GNN.Dataloaders.QM9Dataloader import QM9Dataloader
from En_GNN.models.EnGNNModelLit import EnGNNModelLit
from En_GNN.models.inner_models.EnGNN import EnGNN


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    datamodule = QM9Dataloader(data_dir="./Datasets/QM9/", splits=[130831 - 31000, 18000, 13000], batch_size=cfg.batch_size)
    model = EnGNN(n_input_features=11, n_hidden_features=128, n_output_features=19)
    model_lit = EnGNNModelLit(model, lr=cfg.lr)

    pl_logger = TensorBoardLogger(cfg.logs_path, name="my_model")

    trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                         logger=pl_logger)

    if cfg.train:
        trainer.fit(model_lit, datamodule=datamodule)

    if cfg.test:
        trainer.test(model_lit, datamodule=datamodule)
if __name__ == '__main__':
    main()