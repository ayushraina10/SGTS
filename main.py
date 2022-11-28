# Set a seed value
seed_value= 0
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ['CUDA_LAUNCH_BLOCKING']='1'

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1,2,3'
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
#4. Set pytorch seed
import torch
torch.manual_seed(seed_value)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
################################################################

import argparse
import multiprocessing
import scipy.io as sio
# import os

from Tree.WU_UCT import WU_UCT
from Tree.UCT import UCT

import time

def main():
    parser = argparse.ArgumentParser(description = "SGTS")
    parser.add_argument("--model", type = str, default = "WU-UCT",
                        help = "Base MCTS model WU-UCT/UCT (default: WU-UCT)")

    parser.add_argument("--env-name", type = str, default = "Truss",
                        help = "Environment name")

    parser.add_argument("--MCTS-max-steps", type = int, default = 128,
                        help = "Max simulation step of MCTS (default: 500)")
    parser.add_argument("--MCTS-max-depth", type = int, default = 100,
                        help = "Max depth of MCTS simulation (default: 100)")
    parser.add_argument("--MCTS-max-width", type = int, default = 20,
                        help = "Max width of MCTS simulation (default: 20)")


    parser.add_argument("--gamma", type = float, default = 0.95,
                        help = "Discount factor (default: 1.0)")

    parser.add_argument("--expansion-worker-num", type = int, default = 1,
                        help = "Number of expansion workers (default: 1)")
    parser.add_argument("--simulation-worker-num", type = int, default = 16,
                        help = "Number of simulation workers (default: 16)")

    parser.add_argument("--seed", type = int, default = 0,
                        help = "random seed (default: 0)")

    parser.add_argument("--max-episode-length", type = int, default = 100000,
                        help = "Maximum episode length (default: 100000)")

    parser.add_argument("--policy", type = str, default = "Random",
                        help = "Prior prob/simulation policy used in MCTS Random/PPO/DistillPPO (default: Random)")

    parser.add_argument("--device", type = str, default = "cpu",
                        help = "PyTorch device, if entered 'cuda', use cuda device parallelization (default: cpu)")

    parser.add_argument("--record-video", default = False, action = "store_true",
                        help = "Record video if supported (default: False)")

    parser.add_argument("--mode", type = str, default = "MCTS",
                        help = "Mode MCTS/Distill (default: MCTS)")

    parser.add_argument("--runid", type = str, default = "0",
                        help = "Run identifier")
    
    parser.add_argument("--scenario", type = str, default = "21",
                        help = "Boundary Condition")
    
    parser.add_argument("--trained", type = bool, default = True,
                        help = "Boundary Condition")
    
    parser.add_argument("--repeat", type = int, default = 10,
                        help = "Repeat the episode")

    args = parser.parse_args()

    env_params = {
        "env_name": args.env_name,
        "max_episode_length": args.max_episode_length,
        "random_start": 0,
        "env_type": 'complex',
        "scenario": int(args.scenario)
    }
    
    if args.policy == 'TrussDSNPre':
        args.trained = True
        path = 'Results/final_Pretrained/'
    elif args.policy == 'TrussDSN':
        args.trained = False
        path = 'Results/final_Untrained/'
    elif args.policy == 'TrussDSNcomb':
        args.trained = False
        path = 'Results/final_combTrained/'
    
    print(path)
    import glob
    files_num = len(glob.glob(path+'*.npy'))

    if args.mode == "MCTS":
        # Model initialization
        iterations = args.repeat
        
        init_seeds = np.arange(iterations)
        
        store_states, store_actions, store_rewards, store_times = [], [], np.zeros((iterations, args.max_episode_length)), np.zeros((iterations, args.max_episode_length))
        
        random_num = str(time.time())[-3:]
        
        for ctr, seed in enumerate(init_seeds):
            start_time = time.time()
            
            print(f"Iteration number {ctr} of {args.repeat}")
            env_params['seed'] = seed
            
            MCTStree = WU_UCT(env_params, args.MCTS_max_steps, args.MCTS_max_depth,
                                  args.MCTS_max_width, args.gamma, args.expansion_worker_num,
                                  args.simulation_worker_num, policy = args.policy,
                                  seed = seed, device = args.device,
                                  record_video = args.record_video, args = args)
            #simulate a whole trajectory
            rewards, times, states, actions = MCTStree.simulate_trajectory()
            store_states.append(states)
            store_actions.append(actions)
            store_rewards[ctr, :len(rewards)] = rewards
            store_times[ctr, :len(times)] = times

            MCTStree.close()
            np.save(f'{path}State-action-rewardspretained_{random_num}_{files_num}.npy', [store_states, store_actions, store_rewards, store_times, args])
            end_time = time.time()
            print(f"Iteration number {ctr} of {args.repeat} took {end_time-start_time} seconds")

        # np.save(f'Results/Pretrained/State-action-rewardspretained_{args.runid}.npy', [store_states, store_actions, store_rewards, store_times, args])

if __name__ == "__main__":
    # Mandatory for Unix/Darwin
    multiprocessing.set_start_method("forkserver")
    main()
