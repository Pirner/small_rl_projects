import gym
from tensorflow.keras.models import load_model
import numpy as np

from loss import masked_huber_loss


def main():
    print('run lunar lander')
    model_path = r'D:\rl_stuff\lunar_lander\02_06_2022\model_600.h5'
    model = load_model(
        model_path,
        custom_objects={'masked_huber_loss': masked_huber_loss(0.0, 1.0)},
    )
    max_steps = 1000

    for i in range(10):

        env = gym.make("LunarLander-v2")
        state = env.reset()
        fraction_finished = 0.0
        state = np.append(state, fraction_finished)

        for step in range(max_steps):
            env.render()
            x_in = state[np.newaxis, ...]
            q_values = model.predict(x_in)[0]
            best_action = np.argmax(q_values)
            new_state, reward, done, info = env.step(action=best_action)
            fraction_finished = (step + 1) / max_steps
            new_state = np.append(new_state, fraction_finished)

            state = new_state
            if done:
                break


if __name__ == '__main__':
    main()
