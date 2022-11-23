import numpy as np
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
import torch
from typing import *

from promasens.enums.joint_nn_mode import JointNNMode
from promasens.modules.data_loading.custom_data_module import CustomDataModule
from promasens.modules.neural_networks.lightning_neural_network_wrapper import LightningNeuralNetworkWrapper

# set default tensor type and device
torch.set_default_tensor_type('torch.FloatTensor')

# kinematic parametrization: either "cc" or "ac"
kinematic_parametrization = "cc"

# path to datasets
dataset_type = 'real-world'
if dataset_type == 'analytical':
    if kinematic_parametrization == "cc":
        # train_dataset_name = "analytical_db_n_b-1_n_s-3_n_m-1_T0_n_t-120000_rand_phi_off_rand_psi_s_rand_d_s_r"
        # test_dataset_name = "analytical_db_n_b-1_n_s-3_n_m-1_T3_n_t-400"
        # train_dataset_name = "analytical_db_n_b-2_n_s-6_n_m-2_T0_n_t-120000_rand_phi_off_rand_psi_s_rand_d_s_r"
        # test_dataset_name = "analytical_db_n_b-2_n_s-6_n_m-2_T3_n_t-400"
        train_dataset_name = "analytical_db_n_b-3_n_s-9_n_m-3_T0_n_t-120000_rand_phi_off_rand_psi_s_rand_d_s_r"
        test_dataset_name = "analytical_db_n_b-3_n_s-9_n_m-3_T3_n_t-400_d_s_r_16mm"
        # train_dataset_name = "analytical_db_n_b-3_n_s-9_n_m-3_T0_n_t-120000_random_emf_rand_phi_off"
        # test_dataset_name = "analytical_db_n_b-3_n_s-9_n_m-3_T3_n_t-400_emf_(1.0,0.0,0.0)"
    elif kinematic_parametrization == "ac":
        train_dataset_name = "analytical_db_ac_n_b-1_n_s-9_n_m-2_T0_n_t-120000_rand_phi_off"
        test_dataset_name = "analytical_db_ac_n_b-1_n_s-9_n_m-2_T5_n_t-400"
    else:
        raise NotImplementedError
    
    train_dataset_path = f"datasets/analytical_simulation/{train_dataset_name}.csv"
    test_dataset_path = f"datasets/analytical_simulation/{test_dataset_name}.csv"
else:
    train_dataset_name = f"2022-05-02_FLOWER_SLOW_NOMINAL_P0_R1"
    test_dataset_name = train_dataset_name
    train_dataset_path = f"datasets/experimental/merged_data/{train_dataset_name}_train.csv"
    test_dataset_path = f"datasets/experimental/merged_data/{test_dataset_name}_test.csv"

train_df = pd.read_csv(train_dataset_path).dropna(axis=0)
if dataset_type == "real-world":
    train_df["segment_id"] = 1

num_segments = len(train_df["segment_id"].unique())
num_sensors = len(train_df["sensor_id"].unique())

# training settings
mode = 'train'  # 'train' or 'eval'
seeds = [0, 1, 2]

# hyperparams
batch_size = 650
max_epochs = 250
# manual application of superposition principle for magnetic field
manual_superposition = False
# dfeault nn class is P
from promasens.modules.neural_networks.arch_P import NetNiet as NetNietP
nn_class = NetNietP
if dataset_type == 'analytical':
    if kinematic_parametrization == "cc":
        joint_nn_mode = JointNNMode.EACH_SEGMENT
        lr = 0.18
    elif kinematic_parametrization == "ac":
        from promasens.modules.neural_networks.arch_S import NetNiet as NetNietS
        nn_class = NetNietS
        joint_nn_mode = JointNNMode.EACH_SEGMENT
        lr = 0.01
    else:
        raise NotImplementedError
else:
    joint_nn_mode = JointNNMode.EACH_SENSOR  # separate neural network for each sensor
    lr = 0.00005


def stochastic_statistics_from_dicts(list_of_dicts: List[dict]) -> Dict:
    result_dict = {}
    for key, value in list_of_dicts[0].items():
        values = []
        for single_dict in list_of_dicts:
            values.append(single_dict[key])
        values = np.array(values)
        result_dict[f"{key}_mean"] = np.mean(values)
        result_dict[f"{key}_stdev"] = np.std(values)

    return result_dict


def mean_from_dicts(list_of_dicts: List[dict]) -> Dict:
    result_dict = {}
    for key, value in list_of_dicts[0].items():
        values = []
        for single_dict in list_of_dicts:
            values.append(single_dict[key])
        values = np.array(values)
        result_dict[key] = np.mean(values)

    return result_dict


def train(seeds: List[int]) -> Tuple[Dict, Dict]:
    val_dicts, test_dicts = [], []
    for seed in seeds:
        val_dict, test_dict = train_seed(seed)
        val_dicts.append(val_dict), test_dicts.append(test_dict)

    val_dict = stochastic_statistics_from_dicts(val_dicts)
    test_dict = stochastic_statistics_from_dicts(test_dicts)

    print(f"Validation stochastic statistics:\n", val_dict)
    print(f"Testing stochastic statistics:\n", test_dict)

    return val_dict, test_dict


def train_seed(seed: int) -> Tuple[Dict, Dict]:
    if joint_nn_mode == JointNNMode.ALL:
        return train_model(seed)
    else:
        val_dicts, test_dicts = [], []

        if joint_nn_mode == JointNNMode.EACH_SENSOR:
            for sensor_id in range(num_sensors):
                val_dict, test_dict = train_model(seed, sensor_id=sensor_id)
                val_dicts.append(val_dict), test_dicts.append(test_dict)
        elif joint_nn_mode == JointNNMode.EACH_SEGMENT:
            for segment_id in range(1, num_segments+1):
                val_dict, test_dict = train_model(seed, segment_id=segment_id)
                val_dicts.append(val_dict), test_dicts.append(test_dict)
        else:
            raise NotImplementedError

        val_dict = mean_from_dicts(val_dicts)
        test_dict = mean_from_dicts(test_dicts)

        return val_dict, test_dict


def train_model(seed: int, segment_id: int = None, sensor_id: int = None) -> Tuple[Dict, Dict]:
    # cli = LightningCLI(LightningNeuralNetworkWrapper, CustomDataModule,
    #                    seed_everything_default=0, save_config_overwrite=True, run=False)
    # cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    # cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    # predictions = cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)

    if sensor_id is not None:
        print(f"Training seed {seed} for sensor {sensor_id}.")
    elif segment_id is not None:
        print(f"Training seed {seed} for segment {segment_id}.")
    else:
        print(f"Training seed {seed} for all sensors.")

    seed_everything(seed, workers=True)

    datamodule = CustomDataModule(train_dataset_path, test_dataset_path, segment_id=segment_id, sensor_id=sensor_id,
                                  batch_size=batch_size, num_workers=0, manual_superposition=manual_superposition)
    joint_nn_str_snippet = ""
    if segment_id is not None:
        joint_nn_str_snippet += f"b{segment_id}_"
    if sensor_id is not None:
        joint_nn_str_snippet += f"s{sensor_id}_"

    manual_super_snippet = f"mansuper_" if manual_superposition is True else ""
    model_name = f"db_{train_dataset_name}_{manual_super_snippet}{joint_nn_str_snippet}seed_{seed}"
    pl_model_path = f"statedicts/pl_statedict_{model_name}.ckpt"
    torchscript_model_path = f"statedicts/torchscript_pl_{model_name}.pt"

    if mode == 'train':
        print(f"Initializing new model with input dim {datamodule.input_dim}.")
        nn_params = {"state_dim": datamodule.input_dim}
        model = LightningNeuralNetworkWrapper(nn_class, nn_params, datamodule.train_target_range,
                                              datamodule.val_target_range, datamodule.test_target_range,
                                              lr=lr, manual_superposition=manual_superposition,
                                              output_normalization=False, train_target_mean=datamodule.train_target_mean)
    elif mode == 'eval':
        print(f"Loading model from {pl_model_path}.")
        model = LightningNeuralNetworkWrapper.load_from_checkpoint(pl_model_path)
    else:
        raise ValueError

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    swa_callback = StochasticWeightAveraging(swa_lrs=lr, swa_epoch_start=0.5, annealing_epochs=max_epochs)
    trainer = Trainer(accelerator="auto", max_epochs=max_epochs, log_every_n_steps=1, default_root_dir=f"pl_logs/{model_name}",
                      deterministic=True, detect_anomaly=True,
                      callbacks=[checkpoint_callback, swa_callback])

    if mode == 'train':
        trainer.fit(model, datamodule=datamodule)
        trainer.model.load_from_checkpoint(checkpoint_callback.best_model_path)
        trainer.save_checkpoint(pl_model_path)
        # torch.save(model.nn.state_dict(), torch_model_path)
        script = trainer.model.to_torchscript(file_path=torchscript_model_path, method="script")

    val_dict = trainer.validate(model, datamodule=datamodule)[0]
    test_dict = trainer.test(model, datamodule=datamodule)[0]

    return val_dict, test_dict


if __name__ == "__main__":
    train(seeds)
