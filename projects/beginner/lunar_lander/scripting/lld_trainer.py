import shutil
import uuid
from typing import Tuple

from tensorflow.python.keras.models import load_model
from tqdm import tqdm
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense
import gym

from DTO import StateTransition
from replay import ReplayBuffer
from reward import AverageRewardTracker
from loss import masked_huber_loss


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
        self._replay_buffer_size = 200000
        self._target_network_replace_frequency_steps = 1000
        self._step_count = 0
        self._replay_buffer = ReplayBuffer(self._replay_buffer_size)
        self._training_start = 256
        self._train_every_x_steps = 4
        self._training_batch_size = 128
        self._discount_factor = 0.99
        self._epsilon_decay_factor_per_episode = 0.995
        self._starting_epsilon = 1.0
        self._epsilon = self._starting_epsilon
        self._model_backup_frequency_episodes = 25
        self._minimum_epsilon = 0.01

        self._r_tracker = AverageRewardTracker()
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
        self._discount_factor = 0.99
        self._epsilon = self._starting_epsilon

        for episode in tqdm(range(self._max_episodes), total=self._max_episodes):
            episode_reward = self._run_episode()

            if episode != 0 and episode % self._model_backup_frequency_episodes == 0:
                backup_file = f"model_{episode}.h5"
                print(f"Backing up model to {backup_file}")
                self._model.save(backup_file)

            self._epsilon *= self._epsilon_decay_factor_per_episode
            self._epsilon = max(self._minimum_epsilon, self._epsilon)

            self._r_tracker.add(reward=episode_reward)
            average = self._r_tracker.get_average()

            print(f"episode {episode} finished with reward {episode_reward}. Average reward over last 100: {average}")
            print('epsilon is {0}'.format(self._epsilon))
            if average > 200:
                print('average reward bigger 200')
                break

    def _run_episode(self):
        """
        run an episode in the environment
        :return:
        """
        state = self._env.reset()
        episode_reward = 0

        fraction_finished = 0.0
        state = np.append(state, fraction_finished)

        for step in range(1, self._max_steps + 1):
            self._step_count += 1
            q_values_model = LunarLanderTrainer.predict_q_values(model=self._model, state=state)
            action = LunarLanderTrainer.select_action_epsilon_greedy(q_values=q_values_model, epsilon=self._epsilon)

            # if step == 1:
            # print('Q-values: {0}'.format(q_values_model))
            # print('Max Q: {0}'.format(max(q_values_model)))

            new_state, reward, done, info = self._env.step(action)
            fraction_finished = (step + 1) / self._max_steps
            new_state = np.append(new_state, fraction_finished)

            episode_reward += reward

            if step == self._max_steps:
                print('Episode finished fish maximum number of steps')
                done = True

            state_transition = StateTransition(
                old_state=state,
                action=action,
                reward=reward,
                new_state=new_state,
                done=done,
            )
            self._replay_buffer.add(transition=state_transition)
            state = new_state

            if self._step_count % self._target_network_replace_frequency_steps == 0:
                print('Upadting target model')
                self.copy_model()

            if self._replay_buffer.length() >= self._training_start and self._step_count % self._train_every_x_steps == 0:
                batch = self._replay_buffer.get_batch(batch_size=self._training_batch_size)
                targets = self._calculate_target_values(
                    state_transitions=batch,
                    discount_factor=self._discount_factor
                )
                states = np.array([state_transition.old_state for state_transition in batch])
                LunarLanderTrainer.train_model(model=self._model, states=states, targets=targets)

            if done:
                break

        return episode_reward

    def _calculate_target_values(self, state_transitions, discount_factor):
        """

        :param state_transitions:
        :param discount_factor:
        :return:
        """
        states = []
        new_states = []
        for transition in state_transitions:
            states.append(transition.old_state)
            new_states.append(transition.new_state)

        states = np.array(states)
        new_states = np.array(new_states)
        q_val_new_states = LunarLanderTrainer.get_multiple_q_values(model=self._model, states=new_states)
        q_val_new_states_tg_model = LunarLanderTrainer.get_multiple_q_values(
            model=self._target_model,
            states=new_states
        )
        targets = []

        for index, state_transition in enumerate(state_transitions):
            best_action = LunarLanderTrainer.select_action_epsilon_greedy(
                q_values=q_val_new_states[index],
                epsilon=self._epsilon,
            )
            best_action_next_state_q_val = q_val_new_states_tg_model[index][best_action]

            if state_transition.done:
                target_value = state_transition.reward
            else:
                target_value = state_transition.reward + discount_factor * best_action_next_state_q_val

            target_vector = [0, 0, 0, 0]
            target_vector[state_transition.action] = target_value
            targets.append(target_vector)

        return np.array(targets)

    def copy_model(self) -> None:
        backup_file = 'backup_' + str(uuid.uuid4())
        self._model.save(backup_file)
        self._target_model = load_model(backup_file, custom_objects={'masked_huber_loss': masked_huber_loss(0.0, 1.0)})
        # self._target_model = load_model(backup_file)
        shutil.rmtree(backup_file)

    @staticmethod
    def train_model(model, states, targets):
        model.fit(states, targets, epochs=1, batch_size=len(targets), verbose=0)

    @staticmethod
    def select_action_epsilon_greedy(q_values, epsilon):
        random_value = random.uniform(0, 1)
        if random_value < epsilon:
            return random.randint(0, len(q_values) - 1)

        else:
            return np.argmax(q_values)

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
    def get_multiple_q_values(model, states):
        return model.predict(states)

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
        model.compile(optimizer=optimizer, loss=masked_huber_loss(0.0, 1.0))

        return model


if __name__ == '__main__':
    trainer = LunarLanderTrainer()
    trainer.learn_lunar()
