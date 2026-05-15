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
        # agent sees 5 numbers between 0-500
        # -> N/S queue length
        # -> E/W queue length
        # -> current phase
        # -> time spent in phase
        # -> current timestep (step_count) - gives agent explicit time of day awareness
        self.observation_space = gym.spaces.Box(
            low=0, high=500, shape=(5,), dtype=np.float32
        )


        # variables to track state of intersection at a given time
        # can be thought of as memory of simulation
        self.ns_queue = 0           # number of cars waiting in the N/S direction
        self.ew_queue = 0           # number of cars waiting in the E/W direction
        self.current_phase = 0      # which direction has the green light currently
        self.time_in_phase = 0      # how long has the green light been green for
        self.step_count = 0         # how many timesteps have passed in the current episode
        self.min_green_time = 5     # minimum timesteps a light must stay green before switching is allowed
        self.max_queue = 50         # maximum queue length before overflow penalty is applied




    # resets all variables to 0
    # called at the start of every episode
    def reset(self, seed=None):
        self.ns_queue = 0
        self.ew_queue = 0
        self.current_phase = 0
        self.time_in_phase = 0
        self.step_count = 0


        # called so that the RL algorithm knows what the starting environment looks like
        # packages all 5 state var's into a numpy array
        # -> list of numbers that the agent can read
        return np.array([self.ns_queue, self.ew_queue, self.current_phase, self.time_in_phase, self.step_count], dtype=np.float32), {}
    



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
        # -> penalizes agent for switching phases uneccessarily
        switch_penalty = 5 if action != self.current_phase and self.time_in_phase >= self.min_green_time else 0


        # handle phase switch
        # switches if agent picks a different action AND the light has been green for long enough
        # else increments the time in the current phase
        if action != self.current_phase and self.time_in_phase >= self.min_green_time:
            self.current_phase = action
            self.time_in_phase = 0
        else:
            self.time_in_phase += 1



        # calculates reward
        # penalizes agent for either
        # - any waiting car
        # - changing green lights (to prevent rapid switching)
        # - an imbalance in the number of cars on each side (ex: 40 on N/S and 0 on E/W)
        # - the number of cars waiting on one side exceeds the max permitted
        wait = self.ns_queue + self.ew_queue
        imbalance_penalty = abs(self.ns_queue - self.ew_queue) * 0.5
        overflow_penalty = 50 if self.ns_queue > self.max_queue or self.ew_queue > self.max_queue else 0
        reward = -wait - switch_penalty - imbalance_penalty - overflow_penalty 
        self.step_count += 1



        # end episode after 500 timesteps
        done = self.step_count >= 500


        # hands back 5 things to RL algorithm after every step
        # -> the new observation (5 numbers [0-500]: ns_queue, we_queue, current_phase, time_in_phase, step_count)
        # -> the reward the agent just earned
        # -> 'done' boolean - whether the episode is over
        # -> 'False' - required done-like flag called 'truncated' by Gymnasium (not currently needed)
        # -> '{}' empty dictionary required by Gymnasium for techical reasons
        return np.array([self.ns_queue, self.ew_queue, self.current_phase, self.time_in_phase, self.step_count], dtype=np.float32), reward, done, False, {}    


