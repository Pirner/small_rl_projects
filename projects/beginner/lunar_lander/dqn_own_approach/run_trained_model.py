import gym
from tensorflow.keras.models import load_model
import time

from agent import predict_q_values
from agent import select_best_action
from loss import masked_huber_loss


def main():
    model_path = r'D:\rl_stuff\lunar_lander\last_models\model_700.h5'
    model = load_model(
        model_path,
        custom_objects={'masked_huber_loss': masked_huber_loss(0.0, 1.0)},
    )

    for i in range(20):

        env = gym.make("LunarLander-v2")
        state = env.reset()

        while True:
            env.render()
            q_values = predict_q_values(model=model, state=state)
            best_action = select_best_action(q_values)
            new_state, reward, done, info = env.step(action=best_action)
            print(best_action)
            # time.sleep(0.01)
            state = new_state
            if done:
                break


if __name__ == '__main__':
    main()
