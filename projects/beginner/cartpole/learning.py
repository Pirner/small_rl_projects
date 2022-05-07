import gym
import tensorflow as tf
import numpy as np


class CartpoleAI(object):
    def __init__(self) -> None:
        self._env = gym.make('CartPole-v0')
        seed = 42
        self._env.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Small epsilon value for stablizing division operations
        eps = np.finfo(np.float32).eps.item()

    def render_random_games(self, num_games=10, max_episodes=100):
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
                if done:
                    break

    @property
    def env(self):
        return self._env
