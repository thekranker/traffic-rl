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
    def _on_step(self) -> bool:

        # 'self.model.ep_info_buffer' - buffer that stores information about recently completed episodes
        # 'len > 0' ensures that at least one episode has finished before trying to read from it
        if len(self.model.ep_info_buffer) > 0:
            latest = self.model.ep_info_buffer[-1]      # sets the latest episode
            self.episode_rewards.append(latest['r'])    # appends the total reward from the latest episode
            self.episode_lengths.append(latest['l'])    # appends the length from the latest episode
        return True