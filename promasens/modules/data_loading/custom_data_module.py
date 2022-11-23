import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, TensorDataset

from promasens.utils.df_to_tensor_utils import database_df_to_tensors


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset_path: str, test_dataset_path: str,
                 segment_id: int = None, sensor_id: int = None,
                 batch_size: int = 500, num_workers: int = 0, val_split: float = 0.3,
                 manual_superposition: bool = False):
        super().__init__()

        self.save_hyperparameters()

        df_train = pd.read_csv(self.hparams.train_dataset_path)
        df_test = pd.read_csv(self.hparams.test_dataset_path)

        if self.hparams.segment_id is not None:
            df_train = df_train[df_train["segment_id"] == self.hparams.segment_id]
            df_test = df_test[df_test["segment_id"] == self.hparams.segment_id]

        if self.hparams.sensor_id is not None:
            df_train = df_train[df_train["sensor_id"] == self.hparams.sensor_id]
            df_test = df_test[df_test["sensor_id"] == self.hparams.sensor_id]

        _, x_train, y_train = database_df_to_tensors(df_train, separate_sensors=False,
                                                     manual_superposition=self.hparams.manual_superposition)
        _, x_test, y_test = database_df_to_tensors(df_test, separate_sensors=False,
                                                   manual_superposition=self.hparams.manual_superposition)

        print("train u min", y_train.min(), "train u max", y_train.max())
        print("test u min", y_test.min(), "test u max", y_test.max())

        self.train_target_mean = y_train.mean().item()
        self.train_target_range = (y_train.max() - y_train.min()).item()
        self.val_target_range = (y_train.max() - y_train.min()).item()
        self.test_target_range = (y_test.max() - y_test.min()).item()

        self.input_dim = x_train.size(-1)

        train_set = TensorDataset(x_train, y_train)
        self.test_set = TensorDataset(x_test, y_test)

        val_size = int(len(train_set) * self.hparams.val_split)
        train_size = len(train_set) - int(len(train_set) * self.hparams.val_split)
        self.train_set, self.val_set = random_split(train_set, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
