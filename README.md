# Car Simulation

This module simulates a RWD vehicle modeled under bicycle kinematics.

## Code Structure

The file policy_trpo.py trains an agent in the simulation using the TRPO algorithm. This file instantiates an environment defined in environment.py, which uses model.py to calculate new states from states and actions, and renderer.py to render the simulation in a graphical user interface using Matplotlib. Finally, the goal is defined in goal.csv, and obstacles are defined in obstacles.csv. Each line in the csv files denote one location, specified as (x, y, r) where (x, y) is the location of the goal or obstacle, and r is the radius of the specified area.

## Installation Requirements

This project depends on the following:

* [rllab](https://github.com/rll/rllab)

Clone this simulation repository inside rllab's root directory.

## Usage

Run these commands from the rllab's root directory.

### Running an random agent in simulation

```
python scripts/sim_env.py car_simulation.environment --mode random
```

### Training an agent using TRPO

```
python car_simulation/policy_trpo.py
```
