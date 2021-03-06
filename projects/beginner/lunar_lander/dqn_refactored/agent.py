import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense

from loss import masked_huber_loss


class LunarLanderAgent:
    def __init__(self):
        self._input_shape = (9,)  # 8 variables in the environment + the fraction finished we add ourselves
        self._outputs = 4

    def create_model(self, lr: float, regularization_factor: float):
        model = tf.keras.models.Sequential([
            Dense(64, input_shape=self._input_shape, activation='relu', kernel_regularizer=l2(regularization_factor)),
            Dense(64, activation='relu', kernel_regularizer=l2(regularization_factor)),
            Dense(64, activation='relu', kernel_regularizer=l2(regularization_factor)),
            Dense(self._outputs, activation='linear', kernel_regularizer=l2(regularization_factor))
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer, loss=masked_huber_loss(0.0, 1.0))

        return model

    @staticmethod
    def get_q_values(model, state):
        x_in = state[np.newaxis, ...]
        return model.predict(x_in)[0]

    @staticmethod
    def get_multiple_q_values(model, states):
        return model.predict(states)

    @staticmethod
    def select_action_epsilon_greedy(q_values, epsilon):
        random_value = random.uniform(0, 1)
        if random_value < epsilon:
            return random.randint(0, len(q_values) - 1)

        else:
            return np.argmax(q_values)

    @staticmethod
    def select_best_action(q_values):
        return np.argmax(q_values)
