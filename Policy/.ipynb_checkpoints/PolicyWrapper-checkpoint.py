import numpy as np
import random
import os

# from Policy.PPO.PPOPolicy import PPOAtariCNN, PPOSmallAtariCNN
from Policy.Truss.DSNPolicy import DSN


class PolicyWrapper():
    def __init__(self, policy_name, env_name, action_n, device, env_params = None, max_width = 10):
        self.policy_name = policy_name
        self.env_name = env_name
        self.action_n = action_n
        self.device = device

        self.env_params = env_params
        self.policy_func = None

        self.max_width = max_width

        self.init_policy()

    def init_policy(self):
        if self.policy_name == "Random":
            self.policy_func = None

        elif self.policy_name == "PPO":
            assert os.path.exists("./Policy/PPO/PolicyFiles/PPO_" + self.env_name + ".pt"), "Policy file not found"

            self.policy_func = PPOAtariCNN(
                self.action_n,
                device = self.device,
                checkpoint_dir = "./Policy/PPO/PolicyFiles/PPO_" + self.env_name + ".pt"
            )

        elif self.policy_name == "DistillPPO":
            assert os.path.exists("./Policy/PPO/PolicyFiles/PPO_" + self.env_name + ".pt"), "Policy file not found"
            assert os.path.exists("./Policy/PPO/PolicyFiles/SmallPPO_" + self.env_name + ".pt"), "Policy file not found"

            full_policy = PPOAtariCNN(
                self.action_n,
                device = "cpu", # To save memory
                checkpoint_dir = "./Policy/PPO/PolicyFiles/PPO_" + self.env_name + ".pt"
            )

            small_policy = PPOSmallAtariCNN(
                self.action_n,
                device = self.device,
                checkpoint_dir = "./Policy/PPO/PolicyFiles/SmallPPO_" + self.env_name + ".pt"
            )

            self.policy_func = [full_policy, small_policy]
            
        elif self.policy_name == "TrussDSN":
            checkpoint_dir = ""#, "Policy file not found"
            # assert os.path.exists("./Policy/Truss/PolicyFiles/DSN_" + self.env_name + "_spa.pt")#, "Policy file not found"
            # assert os.path.exists("./Policy/Truss/PolicyFiles/DSN_" + self.env_name + "_sel.pt")#, "Policy file not found"

            self.policy_func = DSN(
                env_params = self.env_params,
                device = self.device,
                checkpoint_dir = checkpoint_dir
            )

        elif self.policy_name == "TrussDSNPre":
            if os.path.exists("./Policy/Truss/PolicyFiles/DSN_enc.pt"):
                checkpoint_dir ="./Policy/Truss/PolicyFiles/DSN" 
            else:
                checkpoint_dir = ""#, "Policy file not found"
            # assert os.path.exists("./Policy/Truss/PolicyFiles/DSN_" + self.env_name + "_spa.pt")#, "Policy file not found"
            # assert os.path.exists("./Policy/Truss/PolicyFiles/DSN_" + self.env_name + "_sel.pt")#, "Policy file not found"

            self.policy_func = DSN(
                env_params = self.env_params,
                device = self.device,
                checkpoint_dir = checkpoint_dir
            )
            
        elif self.policy_name == "TrussDSNcomb":
            if os.path.exists("./Policy/Truss/PolicyFiles/comb_DSN_enc.pt"):
                checkpoint_dir ="./Policy/Truss/PolicyFiles/comb_DSN" 
            else:
                checkpoint_dir = ""#, "Policy file not found"
            # assert os.path.exists("./Policy/Truss/PolicyFiles/DSN_" + self.env_name + "_spa.pt")#, "Policy file not found"
            # assert os.path.exists("./Policy/Truss/PolicyFiles/DSN_" + self.env_name + "_sel.pt")#, "Policy file not found"

            self.policy_func = DSN(
                env_params = self.env_params,
                device = self.device,
                checkpoint_dir = checkpoint_dir
            )
        elif self.policy_name == "TrussDSN_rand":
            if os.path.exists("./Policy/Truss/PolicyFiles/DSN_enc.pt"):
                checkpoint_dir ="./Policy/Truss/PolicyFiles/DSN" 
            else:
                checkpoint_dir = ""
            self.policy_func = DSN(
                env_params = self.env_params,
                device = self.device,
                checkpoint_dir = checkpoint_dir,
                random = True
            )

        else:
            raise NotImplementedError()

    def get_action(self, state):
        if self.policy_name == "Random":
            return random.randint(0, self.action_n - 1)
        elif self.policy_name == "PPO":
            return self.categorical(self.policy_func.get_action(state))
        elif self.policy_name == "DistillPPO":
            return self.categorical(self.policy_func[1].get_action(state))
        elif self.policy_name in ["TrussDSN", "TrussDSNPre", "TrussDSN_rand", "TrussDSNcomb"]:
            return self.categorical_complex(self.policy_func.get_action(state))
        else:
            raise NotImplementedError()

    def get_value(self, state):
        if self.policy_name == "Random":
            return 0.0
        elif self.policy_name == "PPO":
            return self.policy_func.get_value(state)
        elif self.policy_name == "DistillPPO":
            return self.policy_func[0].get_value(state)
        elif self.policy_name in ["TrussDSN", "TrussDSNPre", "TrussDSN_rand", "TrussDSNcomb"]:
            return self.policy_func.get_value(state)
        else:
            raise NotImplementedError()

    def get_prior_prob(self, state):
        if self.policy_name == "Random":
            return np.ones([self.action_n], dtype = np.float32) / self.action_n
        elif self.policy_name == "PPO":
            return self.policy_func.get_action(state)
        elif self.policy_name == "DistillPPO":
            return self.policy_func[0].get_action(state)
        elif self.policy_name in ["TrussDSN", "TrussDSNPre", "TrussDSN_rand", "TrussDSNcomb"]:
            return self.categorical_complex(self.policy_func.get_action(state, explore = False), size =  self.max_width, prior_sample = True)
        else:
            raise NotImplementedError()

    @staticmethod
    def categorical(probs):
        val = random.random()
        chosen_idx = 0

        for prob in probs:
            val -= prob

            if val < 0.0:
                break

            chosen_idx += 1
            
        if chosen_idx >= len(probs):
            chosen_idx = len(probs) - 1
        return chosen_idx
    
    @staticmethod
    def categorical_complex(distribution, size = 1, prior_sample = False):
        probs, actions = distribution
        """Size refers to how many action you want to sample"""
        if len(actions) == 1:
            return [0]
        if max(probs) < 0.01:
            print("MCTS: ALL ZERO ACTIONS: Weird situation")
            print(probs)
            probs = np.ones((len(probs)))/len(probs)
        if size>len(probs):
            size = len(probs)
        if prior_sample:
            #add dirichlet noise
            probs = _dirichlet_noise(probs.copy())
            indexes = np.random.choice(len(probs), p=probs/sum(probs), size = size, replace = False)
            return [probs[indexes], actions[indexes]]  
        return np.random.choice(len(probs), p=probs/sum(probs), size = size, replace = False)

    
def _dirichlet_noise(pi):
    noise_mixing = 1
    epsilon, alpha = 0.01*(noise_mixing) + 0.5*(1-noise_mixing), 0.3#dirichlet_noise parameters (based on chess base alpha zero (0.3), )
    # n = torch.distributions.dirichlet.Dirichlet(torch.tensor([alpha] * len(pi)))        
    # noise = n.sample()
    noise = np.random.dirichlet([alpha] * len(pi))

    return np.array([(1 - epsilon) * p + epsilon * n for p, n in zip(pi, noise)])