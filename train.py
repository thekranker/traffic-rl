from env.traffic_env import TrafficEnv          # import the traffic environment
from stable_baselines3 import DQN               # RL algorithm being used to train the agent
from callbacks import TrainingMetricsCallback   # callback to record episode rewards and lengths during training
import matplotlib.pyplot as plt                 # used for plotting the learning curve
import numpy as np                              # used for math and array utilities


env = TrafficEnv()  # creates an instance of the traffic environment


# creates the DQN agent - three things passed in
# -> 'MlpPolicy' - tells DQN to use the 'Multi-Layer Perceptron' neural network internally
# -> 'env' - the environment the agent train in
# -> 'verbose=1' - tells the algorithm to print training progress to the terminal
model = DQN(
    "MlpPolicy",
    env,
    verbose=1
)


# runs the training of the agent
# note that 5760 timesteps = 1 episode (10,000,000 timesteps = ~1736 episodes)
# callback records episode rewards and lengths throughout training for plotting the learning curve
callback = TrainingMetricsCallback()
model.learn(total_timesteps=10000000, callback=callback)



model.save("dqn_traffic")                   # saves the trained agent to a file named 'dqn_traffic.zip'
print("Training complete. Model saved.")    # confirms in the terminal that training completed successfully



# plots the learning curve - total reward per episode throughout training
# upward trend = agent is learning and improving over time
plt.figure(figsize=(10, 6))             # makes the chart wider and taller for better visibility

# raw episode rewards (faded)
plt.plot(callback.episode_rewards, alpha=0.3, color='gray', label="Raw")

# 100 episode rolling average
plt.plot(np.convolve(callback.episode_rewards, np.ones(100)/100, mode='valid'), color='steelblue', label="Smoothed")

plt.xlabel("Episode")                   # x axis label
plt.ylabel("Total Reward")              # y axis label
plt.title("Agent Learning Curve")       # chart title
plt.legend()                            # draws legend identifying each line
plt.tight_layout()                      # prevents labels from being cut off
plt.savefig("learning_curve.png")       # saves chart as image under 'learning_curve.png'
plt.show()                              # displays chart on screen
print("Learning curve saved.")          # confirms the learning curve was saved successfully

