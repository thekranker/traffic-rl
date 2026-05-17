from stable_baselines3.common.callbacks import BaseCallback     # template to build callback on top of
import numpy as np                                              # used for calculating averages



# callback class that records episode rewards and lengths during training
# called automatically by Stable-Baselines3 at certain points during training
# -> used to plot the learning curve after training is complete
class TrainingMetricsCallback(BaseCallback):


    def __init__(self):

        super().__init__()          # runs Stable-Baselines3 callback setup code before we add ours
        self.episode_rewards = []   # list to store total reward of each episode during training
        self.episode_lengths = []   # list to store how long each episode lasted during training



    # called automatically by Stable-baselines3 every single timestep during training
    # only records data when a new episode has completed to avoid duplicate entries
    # -> compares latest buffer reward to last recorded reward to detect new episodes
    def _on_step(self) -> bool:
        # only record when a new episode has completed
        # checks if buffer has grown since last check to avoid duplicate recording
        if len(self.model.ep_info_buffer) > 0:
            if len(self.episode_rewards) == 0 or self.model.ep_info_buffer[-1]['r'] != self.episode_rewards[-1]:
                latest = self.model.ep_info_buffer[-1]
                self.episode_rewards.append(latest['r'])    # appends total reward from completed episode
                self.episode_lengths.append(latest['l'])    # appends length of completed episode
        return True