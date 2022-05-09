from typing import Tuple, List
import numpy as np
import gym
import time
import tensorflow as tf
import collections
import tqdm

from a2c import Actor, Critic


class LunarLanderAI(object):
    def __init__(self):
        """initialize a LunarLanderAI to train with the a2c method."""
        self._env = gym.make("LunarLander-v2")
        self._actor = Actor(action_space=self._env.action_space.n)
        self._critic = Critic()

    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """move the environment cashed in the class with one action"""
        state, reward, done, _ = self._env.step(action=action)
        return state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32)

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [action], [tf.float32, tf.int32, tf.int32])

    def training_routine(
            self,
            min_episodes_criterion=100,
            max_episodes=10000,
            max_steps_per_episode=1000
    ):
        """"implement method for training the a2c model for lunar lander"""
        reward_threshold = 200
        running_reward = 0
        gamma = 0.99

        episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

        with tqdm.trange(max_episodes) as t:
            for i in t:
                initial_state = self._env.reset()
                while True:
                    self._env.render()

                    action = self._env.action_space.sample()
                    state, reward, done = self.env_step(action=action)

                    if done:
                        break

    def render_random_games(self, num_games=10, max_episodes=1000):
        """
        play a couple of random games and render them
        :param num_games:
        :param max_episodes:
        :return:
        """
        for i in range(num_games):
            self._env.reset()

            for t in range(max_episodes):
                self._env.render()

                action = self._env.action_space.sample()
                next_state, reward, done, info = self._env.step(action)

                print(t, next_state, reward, done, info, action)
                # time.sleep(0.05)
                if done:
                    break
