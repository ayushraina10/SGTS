from multiprocessing import Process
from copy import deepcopy
import random
import numpy as np

from Env.EnvWrapper import EnvWrapper

# from Policy.PPO.PPOPolicy import PPOAtariCNN, PPOSmallAtariCNN

from Policy.PolicyWrapper import PolicyWrapper


# Slave workers
class Worker(Process):
    def __init__(self, pipe, env_params, policy = "Random", gamma = 1.0, seed = 123,
                 device = "cpu", need_policy = True, mcts_params = None):
        super(Worker, self).__init__()

        self.pipe = pipe
        self.env_params = deepcopy(env_params)
        self.gamma = gamma
        self.seed = seed
        self.policy = deepcopy(policy)
        self.device = deepcopy(device)
        self.need_policy = need_policy

        self.wrapped_env = None
        self.action_n = None
        self.max_episode_length = None

        self.policy_wrapper = None
        self.mcts_params = mcts_params

    # Initialize the environment
    def init_process(self):
        self.wrapped_env = EnvWrapper(**self.env_params)

        self.wrapped_env.seed(self.seed)

        self.action_n = self.wrapped_env.get_action_n()
        self.max_episode_length = self.wrapped_env.get_max_episode_length()

    # Initialize the default policy
    def init_policy(self):
        self.policy_wrapper = PolicyWrapper(
            self.policy,
            self.env_params["env_name"],
            self.action_n,
            self.device,
            env_params = self.env_params,
            max_width = self.mcts_params[2]
        )

    def run(self):
        self.init_process()
        self.init_policy()

        # print("> Worker ready.")

        while True:
            # Wait for tasks
            command, args = self.receive_safe_protocol()

            if command == "KillProc":
                return
            elif command == "Expansion":
                checkpoint_data, curr_node, saving_idx, task_idx = args

                # Select expand action, and do expansion
                expand_action, next_state, reward, done, \
                    checkpoint_data = self.expand_node(checkpoint_data, curr_node)

                item = (expand_action, next_state, reward, done, checkpoint_data,
                        saving_idx, task_idx)

                self.send_safe_protocol("ReturnExpansion", item)
            elif command == "Simulation":
                if args is None:
                    raise RuntimeError
                else:
                    task_idx, checkpoint_data, first_action = args
                    state = checkpoint_data[0]

                    # Prior probability is calculated for the new node
                    prior_prob = self.get_prior_prob(state) ##for complex action case, it would be a probability and a set of feasible actions

                    # When simulation invoked because of reaching maximum search depth,
                    # an action was actually selected. Therefore, we need to execute it
                    # first anyway.
                    first_action, actual_action = first_action
                    
                    if first_action is not None:
                        # print(first_action, actual_action)
                        # print("CHECKING IF THE FIRST ACTION MATCHES WITH THE PRIORRRR", first_action, prior_prob[1])
                        # actual_action = prior_prob[1][first_action]
                        # actual_action = prior_prob[1][first_action]
                        
                        state, reward, done = self.wrapped_env.step(actual_action, state = state)

                if first_action is not None and done:
                    accu_reward = reward
                else:
                    # Simulate until termination condition satisfied
                    accu_reward = self.simulate(state)

                if first_action is not None:
                    self.send_safe_protocol("ReturnSimulation", (task_idx, accu_reward, reward, done))
                else:
                    self.send_safe_protocol("ReturnSimulation", (task_idx, accu_reward, prior_prob))

    def expand_node(self, checkpoint_data, curr_node):
        #we dont need to load the environment state
        # self.wrapped_env.restore(checkpoint_data)

        # Choose action to expand, according to the shallow copy node
        # print("about to expand node", curr_node.state[1:], curr_node.prior_prob)
        expand_action = curr_node.select_expand_action()

        # Execute the action, and observe new state, etc.
        # try:
        actual_action = curr_node.prior_prob[1][expand_action]
        # except:
        #     print("some error in indexing", expand_action, len(curr_node.prior_prob[0]), len(curr_node.prior_prob[1]), curr_node.prior_prob, curr_node.state)
        
        next_state, reward, done = self.wrapped_env.step(actual_action, state = curr_node.state.copy())

        if not done:
            checkpoint_data = next_state, next_state[3] #self.wrapped_env.checkpoint()
        else:
            checkpoint_data = None

        return expand_action, next_state, reward, done, checkpoint_data

    def simulate(self, state, max_simulation_step = None, lamda = 0.5):
        step_count = 0
        accu_reward = 0.0
        accu_gamma = 1.0

        start_state_value = self.get_value(state)

        done = False
        rewards = []

        max_simulation_step = self.mcts_params[1] if max_simulation_step == None else max_simulation_step

        # A strict upper bound for simulation count
        while not done and step_count < max_simulation_step:
            # action = self.get_action(state)
            
            #additional calls to the object methods to obtain the set of feasible actions
            pi, feasible_actions = self.policy_wrapper.policy_func.get_action(state)
            action_index = self.policy_wrapper.categorical_complex([pi, feasible_actions])[0]

            next_state, reward, done = self.wrapped_env.step(feasible_actions[action_index], state = state)

            # accu_reward += reward * accu_gamma ##check this and correct it
            # accu_gamma *= self.gamma
            rewards.append(reward)

            state = deepcopy(next_state)

            step_count += 1

        # if not done:
        #     accu_reward += self.get_value(state) * accu_gamma
        if max(rewards)>0.01:
            accu_reward = max(rewards)*(accu_gamma**rewards.index(max(rewards)))
        # Use V(s) to stabilize simulation return
        accu_reward = accu_reward * lamda + start_state_value * (1.0 - lamda)

        return accu_reward

    def get_action(self, state):
        return self.policy_wrapper.get_action(state)

    def get_value(self, state):
        return self.policy_wrapper.get_value(state)

    def get_prior_prob(self, state):
        return self.policy_wrapper.get_prior_prob(state)

    # Send message through pipe
    def send_safe_protocol(self, command, args):
        success = False

        count = 0
        while not success:
            self.pipe.send((command, args))

            ret = self.pipe.recv()
            if ret == command or count >= 10:
                success = True
                
            count += 1

    # Receive message from pipe
    def receive_safe_protocol(self):
        self.pipe.poll(None)

        command, args = self.pipe.recv()

        self.pipe.send(command)

        return deepcopy(command), deepcopy(args)
