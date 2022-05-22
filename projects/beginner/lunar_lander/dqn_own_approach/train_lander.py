import gym
import numpy as np
from tqdm import tqdm

from replay import ReplayBuffer
from agent import create_model, predict_q_values, select_action_epsilon_greedy, copy_model
from agent import get_multiple_q_values, select_best_action
from reward import AverageRewardTracker
from states import StateTransition


def main():
    # replay_buffer_size = 200000
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
    env = gym.make("LunarLander-v2")

    input_shape = (8,)
    output_shape = env.action_space.n

    # set up the training routine
    replay_buffer = ReplayBuffer()
    # create the model which will predict the rewards
    model = create_model(
        lr=learning_rate,
        regularization_factor=regularization_factor,
        outputs=output_shape,
        input_shape=input_shape,
    )
    target_model = copy_model(model)

    # TODO implement target model approach
    epsilon = starting_epsilon
    step_count = 0
    avg_r_tracker = AverageRewardTracker(num_rewards_for_average=100)

    # average = np.inf
    # episode means running a whole simulation -> in the case of the LunarLander so one
    # landing approach in the desired area.
    for episode in tqdm(range(max_episodes), desc='last average reward: {0}, epsilon {1}'.format(
            avg_r_tracker.get_average(), epsilon)):
        # print('Starting episode {0} with epsilon {1}'.format(episode, epsilon))

        episode_reward = 0
        state = env.reset()
        # TODO implement fraction approach for better convergence
        q_values = predict_q_values(model=model, state=state)
        print('Q-values: {0}'.format(q_values))
        print('Max Q: {0}'.format(max(q_values)))

        # run steps in the environment - finish the episode
        for step in range(1, max_steps + 1):
            step_count += 1
            q_values = predict_q_values(model, state)
            action = select_action_epsilon_greedy(q_values=q_values, epsilon=epsilon)
            # env.render()  # in case of optimization do not render the environment

            new_state, reward, done, info = env.step(action=action)
            episode_reward += reward

            if step == max_steps:
                print('episode reached maximum number of steps {0}'.format(max_steps))
                done = True
            state_transition = StateTransition(state, action, reward, new_state, done)
            replay_buffer.add(state_transition)

            state = new_state

            if step_count % target_network_replace_frequency_steps == 0:
                print('Updating target model')
                target_model = copy_model(model)


            if replay_buffer.length() >= training_start and step_count % train_every_x_steps == 0:
                batch = replay_buffer.get_batch(batch_size=training_batch_size)
                # extract the states
                states = [x.old_state for x in batch]
                new_states = [x.new_state for x in batch]

                q_values_states = get_multiple_q_values(model, np.array(states))
                q_values_new_state = get_multiple_q_values(model, np.array(new_states))

                q_values_target_model = get_multiple_q_values(target_model, np.array(states))
                q_values_new_state_target_model = get_multiple_q_values(target_model, np.array(new_states))

                targets = []
                for index, state_transition in enumerate(batch):
                    # best_action = select_best_action(q_values_states[indexx])
                    # best_action_next_state = q_values_new_states[index][best_action]

                    best_action = select_best_action(q_values_new_state[index])
                    best_action_next_state = q_values_new_state_target_model[index][best_action]

                    if state_transition.done:
                        target_value = state_transition.reward
                    else:
                        target_value = state_transition.reward + discount_factor * best_action_next_state

                    target_vector = [0, 0, 0, 0]
                    target_vector[state_transition.action] = target_value
                    targets.append(target_vector)

                targets = np.array(targets)
                model.fit(np.array(states), targets, epochs=1, batch_size=len(targets), verbose=0)

            if done:
                break

        avg_r_tracker.add(episode_reward)
        average = avg_r_tracker.get_average()

        print(f"episode {episode} finished in {step} steps with reward {episode_reward}. Average reward over last 100: {average}")

        # save the model in an interval
        if episode != 0 and episode % model_backup_frequency_episodes == 0:
            backup_file = f"model_{episode}.h5"
            print(f"Backing up model to {backup_file}")
            model.save(backup_file)
        epsilon *= epsilon_decay_factor_per_episode
        epsilon = max(minimum_epsilon, epsilon)

        if average > 200:
            print('average bigger than 200, finished')
            break


if __name__ == '__main__':
    main()
