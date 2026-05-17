import numpy as np          # will handle the math and arrays
import gymnasium as gym     # provides template for environment to follow
import pandas as pd         # used to load the real-world arrival rates CSV



# load real-world arrival rates from Rural Rd & University Dr (Tempe, 2016)
# index_col="hour" allows lookup by hour number (ex: ARRIVAL_RATES.loc[7] = hour 7 rates)
# loaded once at startup as a constant - never changes during training
ARRIVAL_RATES = pd.read_csv("data/arrival_rates.csv", index_col="hour")



# blueprint for the environment
# gym.Env is the 'Gymnasium' template we are building on top of 
# -> class inherits all the structure of a 'Gymnasium' environment
class TrafficEnv(gym.Env):



    def __init__(self, ns_multiplier=1.0, ew_multiplier=1.0):

        super().__init__()  # runs 'Gymnasium' setup code before we add ours


        # defines what the agent is allowed to do
        # 'Discrete(2) indicates exactly 2 choices 
        # - 0 = keep N/S signal green
        # - 1 = keep E/W signal green
        self.action_space = gym.spaces.Discrete(2)


        # defines what the agent can see
        # agent sees 6 numbers
        # - N/S queue length (0-75)
        # - E/W queue length (0-75)
        # - current phase (0 or 1)
        # - time spent in phase
        # - current timestep (step_count) - gives agent explicit time of day awareness (0-5760)
        # - N/S to E/W queue ratio - gives agent explicit relative busyness awareness
        self.observation_space = gym.spaces.Box(
            low=0, high=500, shape=(6,), dtype=np.float32
        )


        # variables to track state of intersection at a given time
        # can be thought of as memory of simulation
        self.ns_queue = 0                   # number of cars waiting in the N/S direction
        self.ew_queue = 0                   # number of cars waiting in the E/W direction
        self.current_phase = 0              # which direction has the green light currently
        self.time_in_phase = 0              # how long has the green light been green for
        self.step_count = 0                 # how many timesteps have passed in the current episode
        self.min_green_time = 1             # minimum timesteps a light must stay green before switching is allowed (1 timestep = 15 sec)
        self.max_queue = 75                 # maximum queue length before overflow penalty is applied
        self.ns_multiplier = ns_multiplier  # scales N/S arrival rates for robustness testing
        self.ew_multiplier = ew_multiplier  # scales E/W arrival rates for robustness testing




    # resets all variables to 0
    # called at the start of every episode
    def reset(self, seed=None):
        self.ns_queue = 0
        self.ew_queue = 0
        self.current_phase = 0
        self.time_in_phase = 0
        self.step_count = 0

        # calculates the ratio of cars in the N/S queue to the E/W queue
        ratio = self.ns_queue / (self.ew_queue + 1)


        # randomizes arrival rate multipliers at the start of each episode
        # -> forces agent to learn a general policy across many traffic conditions
        # -> range 0.5 to 2.0 covers quiet to heavy surge conditions
        self.ns_multiplier = np.random.uniform(0.5, 2.0)
        self.ew_multiplier = np.random.uniform(0.5, 2.0)


        # called so that the RL algorithm knows what the starting environment looks like
        # packages all 6 state var's into a numpy array
        # -> list of numbers that the agent can read
        return np.array([self.ns_queue, self.ew_queue, self.current_phase, self.time_in_phase, self.step_count, ratio], dtype=np.float32), {}
    



    # returns real-world N/S and E/W arrival rates based on current hour of the day
    # uses data from Rural Rd & University Dr (Tempe, 2016) loaded from arrival_rates.csv
    # -> multipliers are randomized each episode during training for adversarial training
    # -> multipliers can be set manually at evaluation time to test specific scenarios
    def _get_arrival_rates(self):
        hour = (self.step_count // 240) % 24    # converts timestep to hour of day (1 timestep = 15 seconds, 240 timesteps = 1 hour)
        ns = ARRIVAL_RATES.loc[hour, 'NS_rate'] # looks up real N/S arrival rate for this hour
        ew = ARRIVAL_RATES.loc[hour, 'EW_rate'] # looks up real E/W arrival rate for this hour
        return ns * self.ns_multiplier, ew * self.ew_multiplier




    # called once every timestep (1 timestep = 15 seconds)
    # 1.) when called, agent passes in an action (0 or 1)
    # 2.) the simulation updates
    # 3.) the environment hands back what happened
    def step(self, action):

        # new cars arrive each timestep
        ns_rate, ew_rate = self._get_arrival_rates()
        self.ns_queue += np.random.poisson(ns_rate)
        self.ew_queue += np.random.poisson(ew_rate)



        # clear cars on the green signal phase
        # if 'current_phase' = 0, we let 6 cars through NS queue each timestep (15 seconds)
        # if 'current_phase' = 1, we let 6 cars through EW queue each timestep (15 seconds)
        # max(0, ...) such that queue never goes negative
        if self.current_phase == 0:
            self.ns_queue = max(0, self.ns_queue - 6)
        else:
            self.ew_queue = max(0, self.ew_queue - 6)


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


        # end episode after 5760 timesteps (5760 x 15 seconds = 24 hours)
        done = self.step_count >= 5760

    
        # calculates the ratio of cars in the N/S queue to the E/W queue
        ratio = self.ns_queue / (self.ew_queue + 1)

        # hands back 5 things to RL algorithm after every step
        # -> the new observation (6 numbers [0-500]: ns_queue, we_queue, current_phase, time_in_phase, step_count, ratio)
        # -> the reward the agent just earned
        # -> 'done' boolean - whether the episode is over
        # -> 'False' - required done-like flag called 'truncated' by Gymnasium (not currently needed)
        # -> '{}' empty dictionary required by Gymnasium for techical reasons
        return np.array([self.ns_queue, self.ew_queue, self.current_phase, self.time_in_phase, self.step_count, ratio], dtype=np.float32), reward, done, False, {}    


