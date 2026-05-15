from env.traffic_env import TrafficEnv  # import the traffic environment
from stable_baselines3 import DQN       # RL algorithm being used to train the agent


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
# tells the agent to interact with the environment for 50,000 timesteps 
# -> aka. 100 complete episodes as defined in 'traffic_env.py'
model.learn(total_timesteps=50000)



model.save("dqn_traffic")                   # saves the trained agent to a file named 'dqn_traffic.zip'
print("Training complete. Model saved.")    # confirms in the terminal that training completed successfully

