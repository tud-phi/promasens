import pytorch_lightning as pl
import torch
from torch import nn


class LightningNeuralNetworkWrapper(pl.LightningModule):
    def __init__(self, nn_class, nn_params,
                 train_target_range: float, val_target_range: float, test_target_range: float,
                 optimizer_class=torch.optim.SGD, lr: float = 0.00001, manual_superposition: bool = False,
                 output_normalization: bool = False, train_target_mean: float = 0.0):
        super().__init__()

        self.save_hyperparameters()

        self.nn = nn_class(**self.hparams.nn_params)
        self.loss_fn = nn.MSELoss(reduction="mean")

        self.train_target_mean = self.hparams.train_target_mean
        self.train_target_range = self.hparams.train_target_range

        self.manual_superposition = self.hparams.manual_superposition
        self.output_normalization = self.hparams.output_normalization

    def forward(self, xi: torch.Tensor):
        if self.manual_superposition:
            u_hat = xi.new_zeros(xi.size(0))
            for k in range(xi.size(1)):
                u_hat_k = self.nn.forward(xi[:, k]).squeeze()
                u_hat = u_hat_k if k == 0 else u_hat + u_hat_k
        else:
            u_hat = self.nn.forward(xi).squeeze()
        if self.output_normalization:
            u_hat = self.train_target_mean + self.train_target_range * u_hat
        return u_hat

    def loss_step(self, batch, batch_idx):
        xi, u = batch
        u_hat = self.forward(xi)
        loss = self.loss_fn(u_hat, u)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.loss_step(batch, batch_idx)
        self.log("train_loss", torch.sqrt(loss), on_epoch=True)
        self.log("test_rel_loss", torch.sqrt(loss) / self.hparams.train_target_range)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = torch.sqrt(self.loss_step(batch, batch_idx))
        self.log("val_loss", loss)
        self.log("val_rel_loss", loss / self.hparams.val_target_range)
        return loss

    def test_step(self, batch, batch_idx):
        loss = torch.sqrt(self.loss_step(batch, batch_idx))
        self.log("test_loss", loss)
        self.log("test_rel_loss", loss / self.hparams.test_target_range)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        if self.hparams.optimizer_class in [torch.optim.Adam, torch.optim.SGD]:
            optimizer = self.hparams.optimizer_class(self.nn.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError

        return optimizer
