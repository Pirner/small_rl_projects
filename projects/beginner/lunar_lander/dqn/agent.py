import tensorflow as tf
import gym
import os
import random

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

import numpy as np
import scipy
import uuid
import shutil

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K

from loss import masked_huber_loss


class DQNLunarLanderAgent:
    def __init__(self):
        self._env = gym.make("LunarLander-v2")

        print(f"Input: {self._env.observation_space.shape[0]}")
        print(f"Output: {self._env.action_space.n}")
        # 8 variables in the environment + the fraction finished we add ourselves
        self._input_shape = (self._env.observation_space.shape[0] + 1,)
        print(self._input_shape)
        self._outputs = 4

        self._optimizer = None
        self._model = None
        self._target_model = None

    def create_model(self, learning_rate, regularization_factor):
        model = Sequential([
            Dense(64, input_shape=self._input_shape, activation="relu", kernel_regularizer=l2(regularization_factor)),
            Dense(64, activation="relu", kernel_regularizer=l2(regularization_factor)),
            Dense(64, activation="relu", kernel_regularizer=l2(regularization_factor)),
            Dense(self._outputs, activation='linear', kernel_regularizer=l2(regularization_factor))
        ])

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._model.compile(optimizer=self._optimizer, loss=masked_huber_loss(0.0, 1.0))

        return model

    def get_q_values(self, model, state):
        inputs = state[np.newaxis, ...]
        return model.predict(inputs)[0]

    def get_multiple_q_values(self, states):
        return self._model.predict(states)

    def get_multiple_q_values_target_model(self, states):
        return self._target_model.predict(states)

    def select_action_epsilon_greedy(self, q_values, epsilon):
        random_value = random.uniform(0, 1)
        if random_value < epsilon:
            return random.randint(0, len(q_values) - 1)
        else:
            return np.argmax(q_values)

    def select_best_action(self, q_values):
        return np.argmax(q_values)

    def copy_model(self, model):
        backup_file = 'backup_' + str(uuid.uuid4())
        model.save(backup_file)
        new_model = load_model(backup_file, custom_objects={'masked_huber_loss': masked_huber_loss(0.0, 1.0)})
        shutil.rmtree(backup_file)
        return new_model

    def train_model(self, model, states, targets):
        model.fit(states, targets, epochs=1, batch_size=len(targets), verbose=0)

    def calculate_target_values(self, model, state_transitions, discount_factor):
        states = []
        new_states = []

        for transition in state_transitions:
            states.append(transition.old_state)
            new_states.append(transition.new_state)

        new_states = np.array(new_states)

        q_values_new_state = self.get_multiple_q_values(states=new_states)
        q_values_new_state_target_model = self.get_multiple_q_values_target_model(states=new_states)

        targets = []
        for index, state_transition in enumerate(state_transitions):
            best_action = self.select_best_action(q_values_new_state[index])
            best_action_next_state_q_value = q_values_new_state_target_model[index][best_action]

            if state_transition.done:
                target_value = state_transition.reward
            else:
                target_value = state_transition.reward + discount_factor * best_action_next_state_q_value

            target_vector = [0] * self._outputs
            target_vector[state_transition.action] = target_value
            targets.append(target_vector)
        return np.array(targets)

    @property
    def env(self):
        return self._env
