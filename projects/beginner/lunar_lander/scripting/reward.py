import numpy as np


class AverageRewardTracker:
    def __init__(self, n=100):
        self._current_index = 0
        self._n = n
        self._last_x_rewards = []

    def add(self, reward):
        if len(self._last_x_rewards) < self._n:
            self._last_x_rewards.append(reward)
        else:
            self._last_x_rewards[self._current_index] = reward
            self.__increment_current_index()

    def __increment_current_index(self):
        self._current_index += 1
        if self._current_index >= self._n:
            self._current_index = 0

    def get_average(self):
        return np.average(self._last_x_rewards)
