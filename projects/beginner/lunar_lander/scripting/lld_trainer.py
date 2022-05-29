from typing import Tuple
from tqdm import tqdm
import numpy as np

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense
import gym


class LunarLanderTrainer:
    def __init__(self):
        self._lr = 0.001
        self._reg_factor = 0.001

        # use fraction finished approach
        self._input_shape = (9,)
        self._env = gym.make("LunarLander-v2")
        self._output_shape = self._env.action_space.n
        self._max_steps = 1000
        self._max_episodes = 10000
        self._step_count = 0

        self._model = LunarLanderTrainer.create_model(
            lr=self._lr,
            reg_factor=self._reg_factor,
            output_shape=self._output_shape,
            input_shape=self._input_shape
        )
        self._target_model = LunarLanderTrainer.create_model(
            lr=self._lr,
            reg_factor=self._reg_factor,
            output_shape=self._output_shape,
            input_shape=self._input_shape
        )

    def learn_lunar(self):
        """

        :return:
        """
        self._main_loop()

    def _main_loop(self):
        """
         runs the mainloop for training and runs through the episodes and stuff
        :return:
        """
        # reset values
        self._step_count = 0

        for episode in tqdm(self._max_episodes, total=self._max_episodes):
            episode_reward = 0
            state = self._env.reset()
            fraction_finished = 0.0
            state = np.append(state, fraction_finished)

            self._run_episode()

    def _run_episode(selfp):
        """
        run an episode in the environment
        :return:
        """
        for step in range(1, self._max_steps + 1):
            self._step_count += 1
            qs_model = LunarLanderTrainer.predict_q_values(self._model)

    @staticmethod
    def predict_q_values(model, state):
        """
        predict q values from given model
        :param model:
        :param state:
        :return:
        """
        x_in = state[np.newaxis, ...]
        return model.predict(x_in)[0]

    @staticmethod
    def create_model(lr: float, reg_factor: float, output_shape: Tuple[int], input_shape: Tuple[int]):
        """
        creates a model instance
        :param lr:
        :param reg_factor:
        :param output_shape:
        :param input_shape:
        :return:
        """
        model = tf.keras.models.Sequential([
            Dense(64, input_shape=input_shape, activation='relu', kernel_regularizer=l2(reg_factor)),
            Dense(64, activation='relu', kernel_regularizer=l2(reg_factor)),
            Dense(64, activation='relu', kernel_regularizer=l2(reg_factor)),
            Dense(output_shape, activation='linear', kernel_regularizer=l2(reg_factor))
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # model.compile(optimizer, loss=masked_huber_loss(0.0, 1.0))
        model.compile(optimizer=optimizer, loss='mse')

        return model


if __name__ == '__main__':
    trainer = LunarLanderTrainer()
