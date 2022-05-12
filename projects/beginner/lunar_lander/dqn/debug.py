import numpy as np

from agent import DQNLunarLanderAgent
from tracking import FileLogger, AverageRewardTracker
from replay_buffer import ReplayBuffer, StateTransition


def main():
    agent = DQNLunarLanderAgent()

    replay_buffer_size = 200000
    learning_rate = 0.001
    regularization_factor = 0.001
    training_batch_size = 128
    training_start = 256
    max_episodes = 10000
    max_steps = 1000
    target_network_replace_frequency_steps = 1000
    model_backup_frequency_episodes = 100
    starting_epsilon = 1.0
    minimum_epsilon = 0.01
    epsilon_decay_factor_per_episode = 0.995
    discount_factor = 0.99
    train_every_x_steps = 4

    replay_buffer = ReplayBuffer(replay_buffer_size)
    model = agent.create_model(learning_rate, regularization_factor)
    target_model = agent.copy_model(model)
    epsilon = starting_epsilon
    step_count = 0
    average_reward_tracker = AverageRewardTracker(100)
    file_logger = FileLogger()

    for episode in range(max_episodes):
        print(f"Starting episode {episode} with epsilon {epsilon}")

        episode_reward = 0
        state = agent.env.reset()
        fraction_finished = 0.0
        state = np.append(state, fraction_finished)

        first_q_values = get_q_values(model, state)
        print(f"Q values: {first_q_values}")
        print(f"Max Q: {max(first_q_values)}")

        for step in range(1, max_steps + 1):
            step_count += 1
            q_values = agent.get_q_values(model, state)
            action = agent.select_action_epsilon_greedy(q_values, epsilon)
            new_state, reward, done, info = agent.env.step(action)

            fraction_finished = (step + 1) / max_steps
            new_state = np.append(new_state, fraction_finished)

            episode_reward += reward

            if step == max_steps:
                print(f"Episode reached the maximum number of steps. {max_steps}")
                done = True

            state_transition = StateTransition(state, action, reward, new_state, done)
            replay_buffer.add(state_transition)

            state = new_state

            if step_count % target_network_replace_frequency_steps == 0:
                print("Updating target model")
                target_model = agent.copy_model(model)

            if replay_buffer.length() >= training_start and step_count % train_every_x_steps == 0:
                batch = replay_buffer.get_batch(batch_size=training_batch_size)
                targets = calculate_target_values(model, target_model, batch, discount_factor)
                states = np.array([state_transition.old_state for state_transition in batch])
                train_model(model, states, targets)

            if done:
                break

        average_reward_tracker.add(episode_reward)
        average = average_reward_tracker.get_average()

        print(
            f"episode {episode} finished in {step} steps with reward {episode_reward}. "
            f"Average reward over last 100: {average}")

        if episode != 0 and episode % model_backup_frequency_episodes == 0:
            backup_file = f"model_{episode}.h5"
            print(f"Backing up model to {backup_file}")
            model.save(backup_file)

        epsilon *= epsilon_decay_factor_per_episode
        epsilon = max(minimum_epsilon, epsilon)


if __name__ == '__main__':
    main()
