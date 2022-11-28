import numpy as np
from copy import deepcopy
import math

from Utils.MovingAvegCalculator import MovingAvegCalculator

# from Policy.PolicyWrapper import PolicyWrapper




# import time
#poli_params = [policy, action_n, device, env_params]

# def get_prior_prob_node(state, policy_params):
#     start = time.process_time()
#     policy, action_n, device, env_params = policy_params
#     policy_wrapper = PolicyWrapper(
#             policy,
#             env_params['env_name'],
#             action_n,
#             device,
#             env_params = env_params
#         )
#     # print("Time to initialize prior", time.process_time() - start)
#     return policy_wrapper.get_prior_prob(state)

class WU_UCTnode():
    def __init__(self, action_n, state, checkpoint_idx, parent, tree,
                 prior_prob = None, is_head = False, env_type = 'complex'):#, policy_params = None):

        # print("new node is being initialized with members:", len(state[1]), "nodes:", len(state[2]), "iteration:", state[3])
        # self.policy_params = policy_params
        if prior_prob == None:
            print("WHYYYYY")
            xxx
            # self.prior_prob = get_prior_prob_node(state, policy_params)
        else:
            self.prior_prob = prior_prob
            # prior = get_prior_prob_node(state, policy_params)
            # print("provided priors", prior_prob, prior, len(prior_prob[0]), len(prior[0]))
            # assert len(prior_prob[0]) == len(prior[0])
            
        self.re_init()
        # #since feasible action state is essential, initialize it firszt
        # self.prior_prob = None
        # if prior_prob is not None:
            # self.re_init()            

        self.env_type = env_type
        self.state = state #similar to compState, think about inclusing string based representation???
        self.checkpoint_idx = checkpoint_idx
        self.parent = parent #what kind of reference is it???
        self.tree = tree #what kind of information is here: object of tree class
        self.is_head = is_head #represents if it is the root node???

        self.visit_count = 0 

        # Record traverse history
        self.traverse_history = dict() #history of the trajectory that led to the design state

        # Visited node count
        self.visited_node_count = 0

        # Updated node count
        self.updated_node_count = 0

        # Moving average calculator
        self.moving_aveg_calculator = MovingAvegCalculator(window_length = 5) 
        ##look into the code for how the average is store???
        if tree is not None:
            self.max_width = tree.max_width #what is the max tree limit???
        else:
            self.max_width = 0

    def re_init(self):
        self.action_n = len(self.prior_prob[0])
        #initialize the values and search stats corresponding to the node

        self.children = [None for _ in range(self.action_n)]
        self.rewards = [0.0 for _ in range(self.action_n)]
        self.dones = [False for _ in range(self.action_n)]
        self.children_visit_count = [0 for _ in range(self.action_n)]
        self.children_completed_visit_count = [0 for _ in range(self.action_n)]
        self.Q_values = [0 for _ in range(self.action_n)]

    def no_child_available(self):
        # All child nodes have not been expanded.
        return self.updated_node_count == 0

    def all_child_visited(self):
        # All child nodes have been visited (not necessarily updated).
        if self.is_head:
            return self.visited_node_count == self.action_n
        else:
            return self.visited_node_count == self.max_width 
            ##check for the relation between the feasible actions and max width???

    def all_child_updated(self):
        # All child nodes have been updated.
        if self.is_head:
            return self.updated_node_count == self.action_n
        else:
            return self.updated_node_count == self.max_width

    # Shallowly clone itself, contains necessary data only.
    def shallow_clone(self):
        action_n, prior_prob = self.action_n, deepcopy(self.prior_prob)
        node = WU_UCTnode(
            action_n = action_n,
            state = deepcopy(self.state),
            checkpoint_idx = self.checkpoint_idx,
            parent = None,
            tree = None,
            prior_prob = prior_prob,
            is_head = False
            # policy_params = self.policy_params
        )

        node.visited_node_count = self.visited_node_count
        node.updated_node_count = self.updated_node_count
        node.max_width = self.max_width

        if self.prior_prob != None:
            node.prior_prob = deepcopy(self.prior_prob)
            node.re_init()
            for action in range(self.action_n):
                if self.children[action] is not None:
                    node.children[action] = 1

            node.children_visit_count = deepcopy(self.children_visit_count)
            node.children_completed_visit_count = deepcopy(self.children_completed_visit_count)

            node.action_n = self.action_n

        return node

    # Select action according to the P-UCT tree policy
    def select_action(self):
        best_score = -10000.0
        best_action = 0
        
        assert self.action_n == len(self.prior_prob[1])
        
        for action in range(self.action_n):
            if self.children[action] is None:
                continue

            # exploit_score = self.Q_values[action] / self.children_completed_visit_count[action]
            #max UCT
            exploit_score = self.Q_values[action]# / self.children_completed_visit_count[action]
            
            
            explore_score = math.sqrt(2.0 * math.log(self.visit_count) / self.children_visit_count[action])
            # explore_score = math.sqrt(2.0 * self.visit_count / self.children_visit_count[action])
            # explore_score = self.prior_prob[0][action]*math.sqrt(2.0 * self.visit_count / self.children_visit_count[action])
            
            score_std = 1#self.moving_aveg_calculator.get_standard_deviation()
            score = exploit_score + score_std * 2.0 * explore_score

            if score > best_score:
                best_score, best_action = score, action

        return best_action

    # Return the action with maximum utility.
    def max_utility_action(self):
        best_score = -10000.0
        best_action = 0

        assert self.action_n == len(self.prior_prob[1])
        
        loops = 0
        
        #max UCT
        best_action = np.argmax(self.Q_values)

#         for action in range(self.action_n):
#             if self.children[action] is None:
#                 continue
#             loops+=1
#             # score = self.Q_values[action] / self.children_completed_visit_count[action]
            
#             #max UCT
#             score = self.Q_values[action]# / self.children_completed_visit_count[action]
            
#             if score > best_score:
#                 best_score = score
#                 best_action = action

        # print("MAX UTILITY ACTION", self.action_n, loops, best_action)
        return best_action

    # Choose an action to expand
    def select_expand_action(self):
        count = 0
        if self.prior_prob == None:
            raise NameError

        while True:
            if count < 20:
                if self.env_type == 'complex':
                    action = int(self.categorical_complex(self.prior_prob))
                else:
                    action = self.categorical(self.prior_prob)
            else:
                action = np.random.randint(0, self.action_n)

                
            if count > 100:
                return action

            if self.children_visit_count[action] > 0 and count < 10:
                count += 1
                continue

            if self.children[action] is None:
                return action

            count += 1

    # Update traverse history, used to perform update
    def update_history(self, idx, action_taken, reward):
        if idx in self.traverse_history:
            return False
        else:
            self.traverse_history[idx] = (action_taken, reward)
            return True

    # Incomplete update, called by WU_UCT.py
    def update_incomplete(self, idx):
        action_taken = self.traverse_history[idx][0]

        if self.children_visit_count[action_taken] == 0:
            self.visited_node_count += 1

        self.children_visit_count[action_taken] += 1
        self.visit_count += 1

    # Complete update, called by WU_UCT.py
    def update_complete(self, idx, accu_reward):
        if idx not in self.traverse_history :
            raise RuntimeError("idx {} should be in traverse_history".format(idx))
        else:
            item = self.traverse_history.pop(idx)
            action_taken = item[0]
            reward = item[1]

        # accu_reward = reward + self.tree.gamma * accu_reward
        accu_reward = max(reward, self.tree.gamma * accu_reward)

        if self.children_completed_visit_count[action_taken] == 0:
            self.updated_node_count += 1

        self.children_completed_visit_count[action_taken] += 1
        # self.Q_values[action_taken] += accu_reward
        #max uct
        self.Q_values[action_taken] = max(accu_reward, self.Q_values[action_taken])
        

        self.moving_aveg_calculator.add_number(accu_reward)

        return accu_reward

    # Add a child to current node.
    def add_child(self, action, child_state, checkpoint_idx, prior_prob = None):#, policy_params = None):
        # print("A node is being added")

        if self.children[action] is not None:
            node = self.children[action]
        else:
            if prior_prob == None:
                print("no priors")
            node = WU_UCTnode(
                action_n = self.action_n,
                state = child_state,
                checkpoint_idx = checkpoint_idx,
                parent = self,
                tree = self.tree,
                prior_prob = prior_prob
                # policy_params = policy_params
            )

            self.children[action] = node

        return node

    # Draw a sample from the categorical distribution parametrized by 'pvals'.
    @staticmethod
    def categorical(pvals):
        num = np.random.random()
        for i in range(pvals.size):
            if num < pvals[i]:
                return i
            else:
                num -= pvals[i]

        return pvals.size - 1

    @staticmethod
    def categorical_complex(distribution, size = 1):
        probs, actions = distribution
        """Size refers to how many action you want to sample"""
        
        if max(probs) < 0.01:
            print("ALL ZERO ACTIONS: Weird situation")
            print(probs)
            probs = np.ones((len(probs)))/len(probs)
        if size>len(probs):
            size = len(probs)
        return np.random.choice(len(probs), p=probs/sum(probs), size = size, replace = False)
