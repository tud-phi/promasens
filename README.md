# Learning 3D Shape Proprioception for Continuum Soft Robots with Multiple Magnetic Sensors
## Abstract
Sensing the shape of continuum soft robots without obstructing their movements and modifying their natural softness requires innovative solutions. 
This letter proposes to use magnetic sensors fully integrated into the robot to achieve proprioception. 
Magnetic sensors are compact, sensitive, and easy to integrate into a soft robot.
We also propose a neural architecture to make sense of the highly nonlinear relationship between the perceived intensity of the magnetic field and the shape of the robot. 
By injecting a priori knowledge from the kinematic model, we obtain an effective yet data-efficient learning strategy. 
We first demonstrate in simulation the value of this kinematic prior by investigating the proprioception behavior when varying the sensor configuration, which does not require us to re-train the neural network. 
We validate our approach in experiments involving one soft segment containing a cylindrical magnet and three magnetoresistive sensors. 
During the experiments, we achieve mean relative errors of 4.5%.

## Paper and Link
Our work is has been published by the Royal Society of Chemistry in Soft Matter.

Please cite our paper if you use our method in your work:
````bibtex
@article{baaij2023learning,
  title={Learning 3D shape proprioception for continuum soft robots with multiple magnetic sensors},
  author={Baaij, Thomas and Holkenborg, Marn Klein and St{\"o}lzle, Maximilian and van der Tuin, Daan and Naaktgeboren, Jonatan and Babu{\v{s}}ka, Robert and Della Santina, Cosimo},
  journal={Soft Matter},
  volume={19},
  number={1},
  pages={44--56},
  year={2023},
  publisher={Royal Society of Chemistry}
}
````

## Instructions

### 1. Prerequisites
This framework requires > **Python 3.8**.

**Note:** To use efficient neural network training, CUDA 11.* needs to be installed and available.

It is recommended to use a package manager like Conda (https://docs.conda.io/en/latest/) to manage the Python version 
and all required Python packages.

### 2. Install git & git-lfs
Please install git and [git-lfs](https://git-lfs.github.com/) to be able to download the repository.

### 3. Clone the repository to your local machine
Clone the repository to your local machine using the following command:
```bash
git clone https://github.com/tud-cor-sr/promasens.git
```

### 4. Installation:
#### 4.1 Install C++ dependencies
Install the cairo graphics library using the instructions on https://www.cairographics.org/download/.
Install on ubuntu:
```bash
sudo apt-get install libcairo2-dev
```
Install on macOS using brew:
```bash
brew install cairo
```

#### 4.2 Install PyTorch
Please install PyTorch according to the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).
The code was tested with PyTorch 1.12.1.
Install using conda with CPU-only support:

```bash
conda install pytorch==1.12.1-c pytorch
```

Install using conda with GPU support:

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

#### 4.3 Install other Python dependencies
Install the remaining Python dependencies using pip:

```bash
pip install . --user
```

### 5. Training of the neural network
The neural networks can be trained using the `trainer_pl.py` script. 
Here, we make use of the PyTorch Lightning framework (https://www.pytorchlightning.ai/).
```bash
python scripts/nn_training/trainer_pl.py
```

### 6. Proprioception
Based on the trained neural networks, inference can be run to estimate the shape of the robot.
For simulated datasets:

```bash
python scripts/inference/infer_simulated_dataset.py
```

For experimental datasets:

```bash
python scripts/inference/infer_experimental_dataset.py
```

### 7. Rendering of the inferred shape sequence
The inferred shape sequence can be rendered using the `visualize_inference.py` script:

```bash
python scripts/visualization/visualize_inference.py
```

## Important notes

### Datasets

#### Simulated datasets
The simulated datasets can be found in the `datasets/analytical_simulation` folder. 
They were generated using the [Magpylib](https://magpylib.readthedocs.io/en/latest/) simulator, which is based on analytical solutions to the magnetic field equations.
Datasets simulating an affine curvature robot, have a _ac_ prefix in their filename.  
All remaining datasets involve the simulation of a Piecewise Constant Curvature (PCC) soft robot.

#### Experimental datasets
The `datasets/experimental` folder contains the experimental datasets in a variety of processing stages:
- `raw_motion_capture_data`: The motion capture data of the tip pose of the robot segment as recorded by the OptiTrack system at 40 Hz.
- `processed_motion_capture_data`: This datasets contains the ground-truth robot configurations obtained by inverse kinematics and also the magnet sensor kinematics evaluated on the ground-truth configurations.
- `sensor_data`: The raw data of the magnetoresistive sensors as recorded by the Arduino at 40 Hz.
- `merged_data`: This dataset contains the merged `processed_motion_capture_data` and `sensor_data` datasets. For this, both datasets are aligned in time.

### Promasens package
The `promasens` package contains the following modules:
- `kinematics`: Contains the kinematic model of the robot: either Constant Curvature (CC), or Affine Curvature (AC). The `SegmentKinematics` class computes the pose of each magnet and sensor using the chosen kinematic parametrization. The `RobotKinematics` class then chains together multiple segments and computes all poses in the inertial / robot base frame. Finally, `SensorMagnetKinematics` computes the proposed parametrization `xi` spatially relating a sensor to each magnet. 
- `neural_network`: Contains the neural network architecture for predicting sensor measurements based on the magnet sensor kinematics `xi`.
- `motion_planning`: Contains several trajectory types in configuration space for a CC and AC segment.
- `simulation`: Contains magnetic field simulations based on [Magpylib](https://magpylib.readthedocs.io/en/latest/) for analytical solutions and [Netgen / NGSolve](https://ngsolve.org) for FEM computations.
- `visualization`: Contains functions for visualizing the loss landscape and rendering the soft robot, the sensors, magnets and magnetic field using PyVista.

### Scripts
Below, we will provide a brief description of most important scripts in the `scripts` folder.
- `scripts/simulated_datasets/gen_simulated_dataset.py`: Generates a simulated dataset using [Magpylib](https://magpylib.readthedocs.io/en/latest/).
- `scripts/experimental_dataset/process_motion_capture_data.py`: Parses the motion capture dataset, runs inverse kinematics, evaluates the magnet sensor kinematics and saves the results to a csv file.
- `scripts/experimental_dataset/merge_sensor_and_motion_capture_data.py`: Merges the motion capture data with the sensor data while first temporarily aligning the data by identifying the initial expansion of the segment.
- `scripts/nn_training/trainer_pl.py`: This script is used to train the neural network.
- `scripts/inference/infer_simulated_dataset.py`: This script is used to infer the shape of the robot for a simulated dataset.
- `scripts/inference/infer_experimental_dataset.py`: This script is used to infer the shape of the robot for an experimental dataset.
- `scripts/visualization/visualize_inference.py`: This script is used to visualize the inferred shape sequence. It uses [Pyvista](https://docs.pyvista.org/) to render the shape of the robot according to the ground-truth (gt) and estimated (hat) configuration.
