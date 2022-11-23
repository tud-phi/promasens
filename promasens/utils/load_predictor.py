import torch

from promasens.enums.joint_nn_mode import JointNNMode
from promasens.modules.neural_networks.lightning_neural_network_wrapper import LightningNeuralNetworkWrapper
from promasens.modules.sensor_measurement_predictor import SensorMeasurementPredictor


def load_predictor(train_dataset_name: str, kinematic_params: dict,
                   num_segments: int, num_sensors: int, seed: int,
                   joint_nn_mode: JointNNMode = JointNNMode.EACH_SENSOR,
                   manual_superposition: bool = False, use_pl_model: bool = True,
                   use_swa: bool = True, device: torch.device = None) -> SensorMeasurementPredictor:
    manual_super_snippet = f"mansuper_" if manual_superposition is True else ""
    if use_pl_model:
        prefix = f"statedicts/pl_statedict_db_{train_dataset_name}_{manual_super_snippet}"
        postfix = f"seed_{seed}.ckpt"
        if joint_nn_mode == JointNNMode.ALL:
            statedict_path = prefix + postfix
            nn = LightningNeuralNetworkWrapper.load_from_checkpoint(statedict_path)
            nn.eval()
            if device is not None:
                nn = nn.to(device)
        else:
            if joint_nn_mode == JointNNMode.EACH_SENSOR:
                id_range = range(num_sensors)
            elif joint_nn_mode == JointNNMode.EACH_SEGMENT:
                id_range = range(1, num_segments+1)
            else:
                raise NotImplementedError

            nn = []
            for id in id_range:
                if joint_nn_mode == JointNNMode.EACH_SENSOR:
                    statedict_path = prefix + f"s{id}_" + postfix
                elif joint_nn_mode == JointNNMode.EACH_SEGMENT:
                    statedict_path = prefix + f"b{id}_" + postfix
                else:
                    raise NotImplementedError

                single_nn = LightningNeuralNetworkWrapper.load_from_checkpoint(statedict_path).to(device)
                single_nn.eval()
                if device is not None:
                    single_nn = single_nn.to(device)
                nn.append(single_nn)
        predictor = SensorMeasurementPredictor(kinematic_params, joint_nn_mode=joint_nn_mode, nn=nn)
    else:
        swa_str = 'swa_' if use_swa else ''
        if joint_nn_mode == JointNNMode.EACH_SENSOR:
            statedict_paths = []
            for sensor_id in range(0, num_sensors):
                statedict_path = f"statedicts/state_dict_model_arch_P_{swa_str}db_{train_dataset_name}_" \
                                 f"{manual_super_snippet}s{sensor_id}_seed_{seed}.pt"
                statedict_paths.append(statedict_path)
        else:
            raise NotImplementedError

        predictor = SensorMeasurementPredictor(kinematic_params)

        # define neural network architecture
        from promasens.modules.neural_networks.arch_P import NetNiet
        predictor.load_torch_nn(nn_statedict_paths=statedict_paths, nn_class=NetNiet, nn_params={}, use_swa=use_swa)

    if device is not None:
        predictor = predictor.to(device)

    return predictor
