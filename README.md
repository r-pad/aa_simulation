# Assured Autonomy - Simulation

This module provides a simulation environment and training scripts for training an RWD vehicle modeled using a kinematic bicycle model with a tire dynamic model to move from one location to another while safely avoiding obstacles. Specifically, this module is tailored for use in an [rllab](https://github.com/rll/rllab) setting, which is a third-party library that we use to train our vehicle planner using deep reinforcement learning.

We train a set of planners to achieve this task. The local planner trains the vehicle to follow circles of arbitrary curvature (with a limit on the max curvature it can follow), which conforms well to Dubins path. The global planner takes in sensor readings and outputs curvatures for the local planner to follow. As of now, only the local planner has been trained.

## Code Structure

The ```envs``` directory contains implementation of simulation environments. Each environment inherits a base environment (in ```base_env.py```), which uses a vehicle model specified in ```model.py```. An optional renderer used viewing trained policies is also implemented in ```renderer.py```.

The ```train``` directory contains training scripts to train local planners and global planners for the vehicle. More information is specified in the files. The training scripts use [rllab](https://github.com/rll/rllab) as a backend.

The ```scripts``` directory contains scripts that may be helpful when evaluating trained policies, exporting trained policies, etc.

## Installation Requirements

This project depends on the following:

* [rllab](https://github.com/rll/rllab)

Clone this simulation repository inside rllab's root directory.

## Usage

Run scripts (such as any of the Python scripts in the directory ```scripts``` or ```train```) from the rllab's root directory, like so:

```
python aa_simulation/{train,scripts}/FILENAME.py
```

Arguments may need to be added to the command depending on the script.
