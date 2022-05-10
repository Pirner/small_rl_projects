import statistics
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
        self._huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self._eps = np.finfo(np.float32).eps.item()

    def compute_loss(
            self,
            action_probs: tf.Tensor,
            values: tf.Tensor,
            returns: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Computes the combined actor-critic loss"""
        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self._huber_loss(values, returns)

        return actor_loss, critic_loss

    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """move the environment cashed in the class with one action"""
        state, reward, done, _ = self._env.step(action=action)
        return state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32)

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [action], [tf.float32, tf.int32, tf.int32])

    def get_expected_return(self, rewards: tf.Tensor, gamma: float, standardize: bool = True) -> tf.Tensor:
        """Compute expected returns per timestep."""
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + self._eps))

        return returns

    def run_episode(self, initial_state: np.ndarray, max_steps: int):
        """run an episode"""
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
            # convert the state into a batched tensor
            state = tf.expand_dims(state, 0)

            # run the model and collect feedback
            # gain action with actor and critic with critic
            action_logits_t = self._actor(state)  # get the action from the actor
            value = self._critic(state)  # reward value from the critic

            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            # store critic values
            action_probs = action_probs.write(t, action_logits_t[0, action])
            values = values.write(t, tf.squeeze(value))

            # Store log probability of the action chosen
            action_probs = action_probs.write(t, action_probs_t[0, action])

            # Apply action to the envirnment to get next state and reward
            state, reward, done = self.tf_env_step(action)
            state.set_shape(initial_state_shape)

            # store reward
            rewards = rewards.write(t, reward)
            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    def train_step(self, initial_state: np.ndarray, gamma: float, max_steps_per_episode: int):
        with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
            # Run the model for one episode to collect training data
            action_probs, values, rewards = self.run_episode(
                initial_state=initial_state,
                max_steps=max_steps_per_episode,
            )

            returns = self.get_expected_return(rewards=rewards, gamma=gamma)
            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]
            ]
            # Calculating loss values to update our network
            actor_loss, critic_loss = self.compute_loss(action_probs=action_probs, values=values, returns=returns)

            # Compute the gradients from the loss
            grads_a = tape_a.gradient(actor_loss, self._actor.trainable_variables)
            grads_c = tape_c.gradient(critic_loss, self._critic.trainable_variables)

            # Apply the gradients to the model's parameters
            self._optimizer.apply_gradients(zip(grads_a, self._actor.trainable_variables))
            self._optimizer.apply_gradients(zip(grads_c, self._critic.trainable_variables))

            episode_reward = tf.math.reduce_sum(rewards)

            return episode_reward

    def training_routine(
            self,
            min_episodes_criterion=100,
            max_episodes=10000,
            max_steps_per_episode=1000
    ):
        """"implement method for training the a2c model for lunar lander"""
        reward_threshold = 200
        reward_threshold = 50
        running_reward = 0
        gamma = 0.99

        episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

        with tqdm.trange(max_episodes) as t:
            for i in t:
                initial_state = self._env.reset()
                # while True:
                # self._env.render()
                episode_reward = int(self.train_step(
                    initial_state=initial_state,
                    gamma=gamma,
                    max_steps_per_episode=max_steps_per_episode,
                ))

                episodes_reward.append(episode_reward)
                running_reward = statistics.mean(episodes_reward)

                t.set_description(f'Episode {i}')
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

                # Show average episode reward every 10 episodes
                if i % 10 == 0:
                    pass  # print(f'Episode {i}: average reward: {avg_reward}')

                if running_reward > reward_threshold and i >= min_episodes_criterion:
                    break
        print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

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
