# SGTS (Sampling Guided Tree Search)
This repository provides the code used in the Self-Learning Design Agent [paper](). Design Strategy Network([paper]())

# Usage


## Prerequisites
- Python 3.x
- PyTorch 1.0
- Gym (with atari) 0.14.0
- Numpy 1.17.2
- Scipy 1.3.1
- OpenCV-Python 4.1.1.26

## Running
1. Download or clone the repository.
2. Run with the default settings:

* A full list of parameters
  * --model: MCTS model to use (currently support WU-UCT and UCT).
  * --env-name: name of the environment.
  * --MCTS-max-steps: number of simulation steps in the planning phase.
  * --MCTS-max-depth: maximum planning depth.
  * --MCTS-max-width: maximum width for each node.
  * --gamma: environment discount factor.
  * --expansion-worker-num: number of expansion workers.
  * --simulation-worker-num: number of simulation workers.
  * --seed: random seed for the environment.
  * --max-episode-length: a strict upper bound of environment's episode length.
  * --policy: default policy (see above).
  * --device: support "cpu", "cuda:x", and "cuda". If entered "cuda", it will use all available cuda devices. Usually used to load the policy.
  #additional parameters for truss
  * --scenario: 
  * --trained:
  * --repeat:

## Run on your own environments (identical to WU-UCT)
We kindly provide an [environment wrapper](https://github.com/liuanji/WU-UCT/tree/master/Env/EnvWrapper.py) and a [policy wrapper](https://github.com/liuanji/WU-UCT/tree/master/Policy/PolicyWrapper.py) to make easy extensions to other environments. All you need is to modify [./Env/EnvWrapper.py](https://github.com/liuanji/WU-UCT/tree/master/Env/EnvWrapper.py) and [./Policy/PolicyWrapper.py](https://github.com/liuanji/WU-UCT/tree/master/Policy/PolicyWrapper.py), and fit in your own environment. Please follow the below instructions.

1. Edit the class EnvWrapper in [./Env/EnvWrapper.py](https://github.com/liuanji/WU-UCT/tree/master/Env/EnvWrapper.py).

    Nest your environment into the wrapper by providing specific functionality in each of the member function of EnvWrapper. There are currently four input arguments to EnvWrapper: *env_name*, *max_episode_length*, *enable_record*, and *record_path*. If additional information needs to be imported, you may first consider adding them in *env_name*.

2. Edit the class PolicyWrapper in [./Policy/PolicyWrapper.py](https://github.com/liuanji/WU-UCT/tree/master/Policy/PolicyWrapper.py).

    Similarly, nest your default policy in PolicyWrapper, and pass the corresponding method using --policy. You will need to rewrite *get_action*, *get_value*, and *get_prior_prob* three member functions.

# Reference
Upcoming publication in IDETC ASME 2022 "Self Learning Design Agent (SLDA): Enabling design learning and tree search in complex action spaces" and submitted to Journal of Mechanical Design 2023 "Learning to design without prior data: Discovering generalizable design strategies using deep learning and tree search" 
