import os
import numpy as np


class AverageRewardTracker():

    def __init__(self, num_rewards_for_average=100):
        self.num_rewards_for_average = num_rewards_for_average
        self.last_x_rewards = []
        self.current_index = 0

    def add(self, reward):
        if len(self.last_x_rewards) < self.num_rewards_for_average:
            self.last_x_rewards.append(reward)
        else:
            self.last_x_rewards[self.current_index] = reward
            self.__increment_current_index()

    def __increment_current_index(self):
        self.current_index += 1
        if self.current_index >= self.num_rewards_for_average:
            self.current_index = 0

    def get_average(self):
        return np.average(self.last_x_rewards)


class FileLogger():

    def __init__(self, file_name='progress.log'):
        self.file_name = file_name
        self.clean_progress_file()

    def log(self, episode, steps, reward, average_reward):
        f = open(self.file_name, 'a+')
        f.write(f"{episode};{steps};{reward};{average_reward}\n")
        f.close()

    def clean_progress_file(self):
        if os.path.exists(self.file_name):
            os.remove(self.file_name)
        f = open(self.file_name, 'a+')
        f.write("episode;steps;reward;average\n")
        f.close()
