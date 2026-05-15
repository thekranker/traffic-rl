import numpy as np          # will handle the math and arrays
import gymnasium as gym     # provides template for environment to follow




# blueprint for the environment
# gym.Env is the 'Gymnasium' template we are building on top of 
# -> class inherits all the structure of a 'Gymnasium' environment
class TrafficEnv(gym.Env):



    def __init__(self):
        super().__init__()  # runs 'Gymnasium' setup code before we add ours


        # defines what the agent is allowed to do
        # 'Discrete(2) indicates exactly 2 choices 
        # -> 0 = keep N/S signal green
        # -> 1 = keep E/W signal green
        self.action_space = gym.spaces.Discrete(2)


        # defines what the agent can see
        # agent sees 4 numbers between 0-100
        # -> N/S queue length
        # -> E/W queue length
        # -> current phase
        # -> time spent in phase
        self.observation_space = gym.spaces.Box(
            low=0, high=100, shape=(4,), dtype=np.float32
        )


        # variables to track state of intersection at a given time
        # can be thought of as memory of simulation
        # -> ns_queue & ew_queue = number of cars waiting in each direction
        # -> current_phase = which direction has green light now
        # -> time_in_phase = how long light has been green for
        # -> step_count = tracks how many timesteps have passed in current episode
        self.ns_queue = 0
        self.ew_queue = 0
        self.current_phase = 0
        self.time_in_phase = 0
        self.step_count = 0




    # resets all variables to 0
    # called at the start of every episode
    def reset(self, seed=None):
        self.ns_queue = 0
        self.ew_queue = 0
        self.current_phase = 0
        self.time_in_phase = 0
        self.step_count = 0


        # called so that the RL algorithm knows what the starting environment looks like
        # packages all 4 state var's into a numpy array
        # -> list of numbers that the agent can read
        return np.array([self.ns_queue, self.ew_queue, self.current_phase, self.time_in_phase], dtype=np.float32), {}
    



    # simulates time-of-day traffic patterns to challenge agent
    # returns N/S arrival rate & E/W arrival rate
    def _get_arrival_rates(self):
        if self.step_count < 100:   # quiet period
            return 1, 1
        elif self.step_count < 200: # morning rush - N/S heavy
            return 5, 1
        elif self.step_count < 300: # midday - moderate and balanced
            return 2, 2
        elif self.step_count < 400: # evening rush - both directions busy
            return 4, 3
        else:                       # night - low traffic
            return 1, 1




    # called once every timestep
    # timestep can be thought of as one second passing at the intersection
    # 1.) when called, agent passes in an action (0 or 1)
    # 2.) the simulation updates
    # 3.) the environment hands back what happened
    def step(self, action):

        # new cars arrive each timestep
        ns_rate, ew_rate = self._get_arrival_rates()
        self.ns_queue += np.random.poisson(ns_rate)
        self.ew_queue += np.random.poisson(ew_rate)



        # clear cars on the green signal phase
        # if 'current_phase' = 0, we let 3 cars through NS queue each timestep
        # if 'current_phase' = 1, we let 3 cars through EW queue each timestep
        # max(0, ...) such that queue never goes negative
        if self.current_phase == 0:
            self.ns_queue = max(0, self.ns_queue - 5)
        else:
            self.ew_queue = max(0, self.ew_queue - 5)


        # calculate switch penalty before phase updates
        switch_penalty = 5 if action != self.current_phase else 0   # penalizes agent for switching phases uneccessarily


        # handle phase switch
        # if agent's selected phase is different from current phase, switch phases
        # else, increment time in current phase
        if action != self.current_phase:    # scenario when agent chooses a different phase than what's currently active
            self.current_phase = action
            self.time_in_phase = 0
        else:
            self.time_in_phase += 1



        # calculate reward
        wait = self.ns_queue + self.ew_queue                        # total number of cars waiting across both directions combined
        reward = -wait - switch_penalty                             # penalizes agent for any waiting car
        self.step_count += 1                                        # ticks clock forward one timestep



        # end episode after 500 timesteps
        done = self.step_count >= 500


        # hands back 5 things to RL algorithm after every step
        # -> the new observation (4 numbers [0-100]: ns_queue, we_queue, current_phase, time_in_phase)
        # -> the reward the agent just earned
        # -> 'done' boolean - whether the episode is over
        # -> 'False' - required done-like flag called 'truncated' by Gymnasium (not currently needed)
        # -> '{}' empty dictionary required by Gymnasium for techical reasons
        return np.array([self.ns_queue, self.ew_queue, self.current_phase, self.time_in_phase], dtype=np.float32), reward, done, False, {}
    


