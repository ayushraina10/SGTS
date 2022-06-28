# import gym
from copy import deepcopy

#import game from the env wrapper
from Env.TrussEnv import Game as Env

# To allow easily extending to other tasks, we built a wrapper on top of the 'real' environment.
class EnvWrapper():
    def __init__(self, env_name, env_type, max_episode_length,\
                     scenario, random_start, seed, enable_record = False, record_path = False): #max_episode_length = 0, enable_record = False, record_path = "1.mp4"):
        self.env_name = env_name
        self.env_type = env_type

        # try:
        # print("intiailizing the environment")
        self.env = Env(max_iterations = max_episode_length,\
                random_start = random_start, scenario = scenario, seed = seed)
        self.recorder = 'hasnt been intialized'
        # Call reset.
        self.env.resetState()
        # except:
        #     xxx
        #     print('Problem loading environment; EnvWrapperTruss.py init')

#         assert isinstance(self.env.action_space, gym.spaces.Discrete), "Should be discrete action space."
        self.action_n = self.env.getActionSize()[0] #50 for trusses, the width can be limited here itself!
        self.max_episode_length = self.env.max_iterations

        self.current_step_count = 0

        self.since_last_reset = 0

    def reset(self):
        state = self.env.resetState()

        self.current_step_count = 0
        self.since_last_reset = 0

        return state

    def step(self, action, state = None):
        next_state, reward, done, _ = self.env.getNextState(action, state = state)

        self.current_step_count = state[3]+1
        if self.current_step_count >= self.max_episode_length:
            done = True

        self.since_last_reset = state[3]+1

        return next_state, reward, done

    def checkpoint(self):
        ##DOUBT make sure the correct state is being updated and hence the copy is correct
        return deepcopy(self.env.get_state()), self.current_step_count

    def restore(self, checkpoint):
        if self.since_last_reset > 20000:
            self.reset()
            self.since_last_reset = 0

        self.restore_full_state(checkpoint[0])

        self.current_step_count = checkpoint[1]

        return self.env.get_state()

    def render(self):
        self.env.render()

    def capture_frame(self):
        print("Check where this is being used and what its utility is")
        raise NotImplementedError
        # self.recorder.capture_frame()

    def store_video_files(self):
        print("Check where this is being used and change appropriately, copy from Coach.py")
        raise NotImplementedError
        # self.recorder.write_metadata()

    def close(self):
        # self.env.close()
        del self.env.env

    def seed(self, seed):
        self.env.env.seed = seed

    def get_action_n(self):
        return self.action_n

    def get_max_episode_length(self):
        return self.max_episode_length

    # def restore_full_state(self, compState):
    #     self.env.env.members, self.env.env.nodes, self.env.env.iteration = compState[1], compState[2], compState[3]