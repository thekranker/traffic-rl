import numpy as np          # will handle the math and arrays
import gymnasium as gym     # provides template for environment to follow
import pandas as pd         # used to load the real-world arrival rates CSV



# real-world arrival rates temporarily disabled - re-enable once agent demonstrates stable learning
# ARRIVAL_RATES = pd.read_csv("data/arrival_rates.csv", index_col="hour")


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
        # - current timestep (step_count) - gives agent explicit time of day awareness (0-500)
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
        self.min_green_time = 10            # minimum timesteps a light must stay green before switching is allowed
        self.max_queue = 75                 # maximum queue length - reserved for future overflow penalty
        self.total_wait_time = 0            # cumulative waiting time across all cars in episode (in timesteps)
        self.total_cars_cleared = 0         # total cars that cleared the intersection in episode
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
        self.total_wait_time = 0
        self.total_cars_cleared = 0

        # calculates the ratio of cars in the N/S queue to the E/W queue
        ratio = self.ns_queue / (self.ew_queue + 1)


        # multipliers fixed at 1.0 for stable base training
        # adversarial randomization will be re-enabled once base policy converges
        self.ns_multiplier = 1.0
        self.ew_multiplier = 1.0


        # called so that the RL algorithm knows what the starting environment looks like
        # packages all 6 state var's into a numpy array
        # -> list of numbers that the agent can read
        return np.array([self.ns_queue, self.ew_queue, self.current_phase, self.time_in_phase, self.step_count, ratio], dtype=np.float32), {}
    



    # simulates time-of-day traffic patterns
    # uses fixed asymmetric rates to give agent a clear pattern to learn
    # real-world data will be re-enabled once agent demonstrates stable learning
    def _get_arrival_rates(self):
        if self.step_count < 100:       # quiet period
            ns, ew = 1, 1
        elif self.step_count < 200:     # morning rush - NS heavy
            ns, ew = 5, 1
        elif self.step_count < 300:     # midday - balanced
            ns, ew = 2, 2
        elif self.step_count < 400:     # evening rush - both busy
            ns, ew = 4, 3
        else:                           # night - low traffic
            ns, ew = 1, 1
        return ns * self.ns_multiplier, ew * self.ew_multiplier




    # called once every timestep
    # 1.) when called, agent passes in an action (0 or 1)
    # 2.) the simulation updates
    # 3.) the environment hands back what happened
    def step(self, action):

        # new cars arrive each timestep
        ns_rate, ew_rate = self._get_arrival_rates()
        self.ns_queue += np.random.poisson(ns_rate)
        self.ew_queue += np.random.poisson(ew_rate)



        # clear cars on the green signal phase and track how many cleared
        # 'cleared' = actual number of cars that passed through (capped at 3 per timestep)
        # uses min() so we can track exact cars cleared
        if self.current_phase == 0:
            cleared = min(self.ns_queue, 3)             # actual number of N/S cars that cleared
            self.ns_queue -= cleared                    # remove cleared cars from N/S queue
            self.total_cars_cleared += cleared          # add to total cars cleared counter
        else:
            cleared = min(self.ew_queue, 3)             # actual number of E/W cars that cleared
            self.ew_queue -= cleared                    # remove cleared cars from E/W queue
            self.total_cars_cleared += cleared          # add to total cars cleared counter



        # handle phase switch
        # switches if agent picks a different action AND the light has been green for long enough
        # else increments the time in the current phase
        if action != self.current_phase and self.time_in_phase >= self.min_green_time:
            self.current_phase = action
            self.time_in_phase = 0
        else:
            self.time_in_phase += 1


        # calculates reward
        # uses negative total queue length as the reward signal
        # simple and direct - agent's only goal is to minimize cars waiting
        reward = -(self.ns_queue + self.ew_queue)               # subtracts every waiting car from agent's reward
        self.total_wait_time += self.ns_queue + self.ew_queue   # every waiting car adds 1 timestep of wait
        self.step_count += 1


        # end episode after 500 timesteps
        done = self.step_count >= 500

    
        # calculates the ratio of cars in the N/S queue to the E/W queue
        ratio = self.ns_queue / (self.ew_queue + 1)

        # hands back 5 things to RL algorithm after every step
        # -> the new observation (6 numbers [0-500]: ns_queue, we_queue, current_phase, time_in_phase, step_count, ratio)
        # -> the reward the agent just earned
        # -> 'done' boolean - whether the episode is over
        # -> 'False' - required done-like flag called 'truncated' by Gymnasium (not currently needed)
        # -> '{}' empty dictionary required by Gymnasium for techical reasons
        return np.array([self.ns_queue, self.ew_queue, self.current_phase, self.time_in_phase, self.step_count, ratio], dtype=np.float32), reward, done, False, {}    


