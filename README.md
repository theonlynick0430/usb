# Cluttered USB Insertion

This repository implements a robotic USB insertion task in a cluttered environment using the Franka Emika Panda robot. The system demonstrates both scripted policies and motion planning approaches for solving the task, with support for different scene configurations and planning parameters.

## Results

The system successfully demonstrates USB insertion in different environments. Below are visualizations of the motion planning policy in various scenarios:

| Cluttered Scene | Spheres Scene |
|----------------|---------------|
| ![Cluttered Scene](assets/mp_clutter.gif) | ![Spheres Scene](assets/mp_spheres.gif) |
| **Wall Scene** | **Plain Scene** |
| ![Wall Scene](assets/mp_wall.gif) | ![Plain Scene](assets/mp_plain.gif) |

## Overview

The system simulates a USB insertion task where a robot must:
1. Grasp a USB plug
2. Navigate through a cluttered environment
3. Insert the plug into a USB port

The implementation includes:
- Scripted policy for basic insertion
- Hybrid RRT-Connect* motion planning for complex environments
- Multiple scene configurations with varying levels of difficulty
- Configurable planning parameters

## Installation

1. Clone the repository:
```bash
git clone https://github.com/theonlynick0430/usb
cd usb
```

2. Create env and install dependencies:
```bash
conda create --name usb python=3.10
conda activate usb
pip install -r requirements.txt
pip install "qpsolvers[quadprog]"
```
## Usage

### Running the Scripted Policy

```bash
python scripted_policy.py
```

### Running the Motion Planning Policy

The motion planning policy can be run with different scene configurations and planning parameters using Hydra. The default configuration is in `configs/config.yaml`.

1. Default configuration (cluttered scene):
```bash
python motion_plan.py
```

2. Specify a different scene:
```bash
python motion_plan.py scene=plain    # No obstacles
python motion_plan.py scene=wall     # Only wall obstacle
python motion_plan.py scene=spheres  # Only floating spheres
python motion_plan.py scene=clutter  # Wall and spheres (default)
```

3. Adjust planning parameters:
```bash
python motion_plan.py step_size=0.1 max_iters=2000
```

You can override any of these parameters via command line arguments.

## Authors

- Nikhil Sridhar 