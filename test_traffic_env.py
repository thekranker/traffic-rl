# Testing / Debugging of 'traffic_env.py

from env.traffic_env import TrafficEnv  # import the traffic environment


if __name__ == "__main__":                  # python convention meaning "only run this code if I'm running this file directly"
    env = TrafficEnv()                      # creates an instance of the traffic environment     
    obs, _ = env.reset()                    # calls reset to get the starting observation
    print(f"Initial observation: {obs}")    # prints the starting observation, you should see "Initial observation: [0. 0. 0. 0.]"

    # loops through 5 steps
    # always passing action 0 (N/S signal stays green)
    # prints the state after each step
    for i in range(5):
        action = 0
        obs, reward, done, _, _ = env.step(action)
        print(f"Step {i+1} | NS: {obs[0]:.0f} | EW: {obs[1]:.0f} | Reward: {reward:.0f}")