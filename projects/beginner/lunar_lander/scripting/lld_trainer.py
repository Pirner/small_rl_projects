from typing import Tuple

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
