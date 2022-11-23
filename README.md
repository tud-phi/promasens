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
Our work is currently under review with Soft Matter.

Please cite our paper if you use our method in your work:
````bibtex
@article{baaij2022learning,
  title={Learning 3D Shape Proprioception for Continuum Soft Robots with Multiple Magnetic Sensors},
  author={Baaij, Thomas and Klein Holkenborg, Marn and Stölzle, Maximilian and van der Tuin, Daan and Naaktgeboren, Jonatan and Babuska, Robert and Della Santina, Cosimo},
  journal={Soft Matter},
  year={2022},
  publisher={Royal Society of Chemistry}
  notes={Under review}
}
````

## Instructions

### 1. Prerequisites
This framework requires > **Python 3.8**. The generation of synthetic datasets requires an Ubuntu environment. 

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

## Important notes
### Datasets
#### Simulated datasets
#### Experimental datasets