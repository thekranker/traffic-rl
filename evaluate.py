from env.traffic_env import TrafficEnv  # import the traffic environment
from stable_baselines3 import DQN       # RL algorithm being used to train the agent
import numpy as np                      # will handle the math and arrays
import matplotlib.pyplot as plt         # used for plotting the comparison results



# runs one complete episode of the traffic simulation & returns the total reward earned
# takes two inputs - the environment & an optional model
def run_episode(env, model=None, fixed_timer=False):

    obs, _ = env.reset()    # resets environment
    total_reward = 0        # resets total reward


    # runs loop for 500 timesteps - aka. one episode
    for _ in range(500):
        if model is not None:                       # if a model is passed in
            action, _ = model.predict(obs)          # model picks an action - "model looks at current state and makes an informed decision"
        elif fixed_timer:                           # if fixed timer is enabled
            action = 1 if env.time_in_phase >= 10 and env.time_in_phase >= env.min_green_time else 0  # switch green lights at fixed timesteps
        else:                                       # if no model is passed in
            action = env.action_space.sample()      # picks a random action (0 or 1) - serves as random baseline controller

        obs, reward, done, _, _ = env.step(action)  # calls the 'step' function in 'traffic_env.py' and unpacks the 5 things it returns
        total_reward += reward                      # total reward accumulates each step

        if done:
            break
            
    return total_reward                             # return total reward




env = TrafficEnv()                          # creates an instance of the traffic environment
model = DQN.load("dqn_traffic", env=env)    # reads the saved 'dqn_traffic.zip' and reconstructs the trained agent


# stores the total reward of each episode for each controller type
#   DQN agent -   trained RL agent
#    random   -   picks action 0 or 1 randomly
#    fixed    -   switches the light on a fixed schedule
# run 20 episodes per controller so comparison is fair and averaged out
dqn_rewards = []
random_rewards = []
fixed_rewards = []
num_episodes = 20



for i in range(num_episodes):                                   # runs 20 episodes for each controller
    dqn_rewards.append(run_episode(env, model=model))           # runs the DQN agent for one episode
    random_rewards.append(run_episode(env, model=None))         # runs the random controller for one episode
    fixed_rewards.append(run_episode(env, fixed_timer=True))    # runs the fixed timer controller for one episode



print(f"DQN Average Reward:        {np.mean(dqn_rewards):.0f}")     # prints the mean DQN agent's reward
print(f"Random Average Reward:      {np.mean(random_rewards):.0f}") # prints the mean random signal reward
print(f"Fixed Timer Average Reward: {np.mean(fixed_rewards):.0f}")  # prints the mean fixed timer reward



# plots the comparison of all three controllers across 20 episodes
episodes = list(range(1, num_episodes + 1))     # forces the chart to use 1-20 values for x axis
plt.figure(figsize=(10, 6))                     # makes the chart wider and taller for better visibility

plt.plot(episodes, dqn_rewards, label="DQN")            # draws a line for DQN agent rewards
plt.plot(episodes, fixed_rewards, label="Fixed Timer")  # draws a line for fixed timer rewards
plt.plot(episodes, random_rewards, label="Random")      # draws a line for random signal rewards

plt.xlabel("Episode")                                   # x axis label
plt.ylabel("Total Reward")                              # y axis label
plt.title("DQN vs Fixed Timer vs Random Controller")    # chart title
plt.legend()                                            # draws legend to identify each line
plt.tight_layout()                                      # prevents labels from being cut off
plt.savefig("results.png")                              # saves chart as image under 'results.png'
plt.show()                                              # display chart on screen