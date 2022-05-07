import collections
import statistics
from typing import Tuple, List
import time

import gym
import tensorflow as tf
import numpy as np
import tqdm


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""
    def __init__(self, num_actions: int, num_hidden_units: int):
        super().__init__()

        self.common = tf.keras.layers.Dense(num_hidden_units, activation="relu")
        self.actor = tf.keras.layers.Dense(num_actions)
        self.critic = tf.keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


class CartpoleAI(object):
    def __init__(self) -> None:
        self._env = gym.make('CartPole-v0')
        # self._env = gym.make("LunarLander-v2")
        seed = 42
        self._env.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Small epsilon value for stablizing division operations
        self._eps = np.finfo(np.float32).eps.item()

        self.num_actions = self._env.action_space.n  # 2
        self.num_hidden_units = 128
        self._model = ActorCritic(num_actions=self.num_actions, num_hidden_units=self.num_hidden_units)
        self._huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


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

    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        returns state, reward and done flag given an action.
        :param action:
        :return:
        """
        state, reward, done, _ = self._env.step(action=action)
        return state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32)

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [action], [tf.float32, tf.int32, tf.int32])

    def run_episode(self, initial_state: tf.Tensor, max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """

        :param initial_state:
        :param max_steps:
        :return:
        """
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities and critic value
            action_logits_t, value = self._model(state)

            # Sample next action from the action probability distribution
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            # Store critic values
            values = values.write(t, tf.squeeze(value))

            # Store log probability of the action chosen
            action_probs = action_probs.write(t, action_probs_t[0, action])

            # Apply action to the environment to get next state and reward
            state, reward, done = self.tf_env_step(action)
            state.set_shape(initial_state_shape)

            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

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

    def compute_loss(self, action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss"""
        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self._huber_loss(values, returns)

        return actor_loss + critic_loss

    # @tf.function
    def train_step(
            self,
            initial_state: tf.Tensor,
            gamma: float,
            max_steps_per_episode: int
    ) -> tf.Tensor:
        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            action_probs, values, rewards = self.run_episode(
                initial_state=initial_state,
                max_steps=max_steps_per_episode
            )

            returns = self.get_expected_return(rewards=rewards, gamma=gamma)
            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]
            ]

            # Calculating loss values to update our network
            loss = self.compute_loss(action_probs=action_probs, values=values, returns=returns)

            # Compute the gradients from the loss
            grads = tape.gradient(loss, self._model.trainable_variables)

            # Apply the gradients to the model's parameters
            self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

            episode_reward = tf.math.reduce_sum(rewards)

            return episode_reward

    def train_model(self, min_episodes_criterion=100, max_episodes=10000, max_steps_per_episode=1000):

        # Cartpole-v0 is considered solved if average reward is >= 195 over 100
        # consecutive trials
        reward_threshold = 195
        # reward_threshold = 50
        running_reward = 0

        gamma = 0.99
        # Keep last episodes reward
        episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

        with tqdm.trange(max_episodes) as t:
            for i in t:
                initial_state = tf.constant(self._env.reset(), dtype=tf.float32)
                episode_reward = int(self.train_step(
                    initial_state=initial_state,
                    gamma=gamma,
                    max_steps_per_episode=max_steps_per_episode))

                episodes_reward.append(episode_reward)
                running_reward = statistics.mean(episodes_reward)

                t.set_description(f'Episode {i}')
                t.set_postfix(
                    episode_reward=episode_reward, running_reward=running_reward)

                # Show average episode reward every 10 episodes
                if i % 10 == 0:
                    pass  # print(f'Episode {i}: average reward: {avg_reward}')

                if running_reward > reward_threshold and i >= min_episodes_criterion:
                    break
        print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

    def render_episode(self, max_steps=100):
        # screen = self._env.render(mode='rgb_array')

        state = tf.constant(self._env.reset(), dtype=tf.float32)
        for i in range(1, max_steps + 1):
            self._env.render()

            state = tf.expand_dims(state, 0)
            action_probs, _ = self._model(state)
            action = np.argmax(np.squeeze(action_probs))

            state, _, done, _ = self._env.step(action)
            state = tf.constant(state, dtype=tf.float32)

            time.sleep(1)

            if done:
                break

    @property
    def env(self):
        return self._env
