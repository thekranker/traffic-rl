from env.traffic_env import TrafficEnv  # import the traffic environment
from stable_baselines3 import PPO       # RL algorithm being used to train the agent
import numpy as np                      # will handle the math and arrays
import matplotlib.pyplot as plt         # used for plotting the comparison results



# runs one complete episode of the traffic simulation & returns the total reward earned
# takes two inputs - the environment & an optional model
def run_episode(env, model=None, fixed_timer=False):

    obs, _ = env.reset()    # resets environment
    total_reward = 0        # resets total reward


    # runs loop for 5760 timesteps - aka. one episode
    for _ in range(5760):
        if model is not None:                       # if a model is passed in
            action, _ = model.predict(obs)          # model picks an action - "model looks at current state and makes an informed decision"
        elif fixed_timer:                           # if fixed timer is enabled
            action = 1 if env.time_in_phase >= 2 and env.time_in_phase >= env.min_green_time else 0  # switch lights at fixed timesteps
        else:                                       # if no model is passed in
            action = env.action_space.sample()      # picks a random action (0 or 1) - serves as random baseline controller

        obs, reward, done, _, _ = env.step(action)  # calls the 'step' function in 'traffic_env.py' and unpacks the 5 things it returns
        total_reward += reward                      # total reward accumulates each step

        if done:
            break
            
    avg_wait_seconds = (env.total_wait_time / max(env.total_cars_cleared, 1)) * 15      # calculate the average wait time of a car
    return total_reward, avg_wait_seconds                                               # returns the total reward & average car wait time



# defines three robustness scenarios to test the agent against
# each scenario has a name and multipliers that scale the arrival rates
# -> Normal   - identical to training conditions, serves as control
# -> NS Surge - doubles N/S arrival rate, simulates a major event on N/S corridor
# -> EW Surge - triples E/W arrival rate, reverses usual pattern the agent was trained on
scenarios = [
    {"name": "Normal",   "ns_multiplier": 1.0, "ew_multiplier": 1.0},
    {"name": "NS Surge", "ns_multiplier": 2.0, "ew_multiplier": 1.0},
    {"name": "EW Surge", "ns_multiplier": 1.0, "ew_multiplier": 3.0},
]



model = PPO.load("ppo_traffic")     # loads the saved trained agent from 'ppo_traffic.zip'
num_episodes = 20                   # the number of episodes run per controller per scenario

results = {}                        # dictionary to store the results for every scenario




# for loop to iterate through each of the three scenarios
# creates three different traffic environments for robustness testing
for scenario in scenarios:
    env = TrafficEnv(
        ns_multiplier=scenario["ns_multiplier"],
        ew_multiplier=scenario["ew_multiplier"]
    )
    
    ppo_rewards = []        # stores PPO agent rewards
    random_rewards = []     # stores random signal rewards
    fixed_rewards = []      # stores fixed timer rewards
    ppo_waits = []          # stores PPO agent average wait times
    random_waits = []       # stores random controller average wait times
    fixed_waits = []        # stores fixed timer average wait times



    # runs 20 episodes for each controller in this scenario
    # stores both total reward and average wait time per car for each episode
    for _ in range(num_episodes):
        reward, wait = run_episode(env, model=model)        # runs PPO agent - gets reward and avg wait
        ppo_rewards.append(reward)                          # stores total reward
        ppo_waits.append(wait)                              # stores avg wait time in seconds

        reward, wait = run_episode(env, model=None)         # runs random controller - gets reward and avg wait
        random_rewards.append(reward)                       # stores total reward
        random_waits.append(wait)                           # stores avg wait time in seconds

        reward, wait = run_episode(env, fixed_timer=True)   # runs fixed timer - gets reward and avg wait
        fixed_rewards.append(reward)                        # stores total reward
        fixed_waits.append(wait)                            # stores avg wait time in seconds



    # stores results for this scenario in the results dictionary
    # -> each controller maps to a dictionary with both rewards and wait times
    results[scenario["name"]] = {
        "PPO":         {"rewards": ppo_rewards,   "waits": ppo_waits},
        "Fixed Timer": {"rewards": fixed_rewards,  "waits": fixed_waits},
        "Random":      {"rewards": random_rewards, "waits": random_waits}
    }



# loops through the results dictionary and prints a summary of each scenario
for scenario_name, controllers in results.items():              # iterates through outer dictionary -> gives scenario name
    print(f"\n--- {scenario_name} ---")                         # prints scenario name header
    for controller_name, data in controllers.items():           # iterates through inner dictionary -> gives controller data
        print(f"{controller_name}: reward={np.mean(data['rewards']):.0f} | avg wait={np.mean(data['waits']):.1f}s")     # prints data




# debug block - runs one episode and records every action the PPO agent takes
# used to understand what strategy the agent has learned
# -> ~50/50 split = agent learned nothing meaningful
# -> heavily skewed = agent learned to favor one direction
# -> frequent switching = agent may be switching too often
env_debug = TrafficEnv()
obs, _ = env_debug.reset()      # reset environment to the start of a new episode
actions = []                    # empty list that holds every action the agent will take throughout the episode
for _ in range(5760):           # one full 24-hour episode
    action, _ = model.predict(obs)                      # agent picks an action
    actions.append(int(action))                         # record the action
    obs, _, done, _, _ = env_debug.step(int(action))    # steps the environment forward with the picked action
    if done:
        break

# prints the percentage of time the agent chose each action
print(f"\nDebug - PPO action distribution:")
print(f"NS green (action 0): {actions.count(0)} times ({actions.count(0)/len(actions)*100:.1f}%)")  # divides actions to get a percentage
print(f"EW green (action 1): {actions.count(1)} times ({actions.count(1)/len(actions)*100:.1f}%)")  # divides actions to get a percentage




fig, axes = plt.subplots(1, 3, figsize=(15, 5))     # creates a figure with 3 side-by-side charts (one for each scenario)

# loops through results and draws one chart per scenario
for i, (scenario_name, controllers) in enumerate(results.items()):
    ax = axes[i]                                    # selects the correct chart slot (0=Normal, 1=NS Surge, 2=EW Surge)
    episodes = list(range(1, num_episodes + 1))     # forces x axis to use 1-20 episode numbers

    # draws a line for each controller in this scenario
    for controller_name, data in controllers.items():
        ax.plot(episodes, data['rewards'], label=controller_name)   # plots rewards for this controller

    ax.set_title(scenario_name)     # sets chart title to scenario name
    ax.set_xlabel("Episode")        # x axis label
    ax.set_ylabel("Total Reward")   # y axis label
    ax.legend()                     # draws legend identifying each controller line


plt.tight_layout()                          # adjusts spacing between charts to prevent overlap
plt.savefig("robustness_results.png")       # saves all three charts as a single image
plt.show()                                  # displays the figure on screen
print("Robustness results saved.")          # confirms the robustness results were saved successfully