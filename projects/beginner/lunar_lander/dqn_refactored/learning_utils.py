import shutil

import numpy as np
import uuid
from tensorflow.keras.models import load_model

from agent import LunarLanderAgent
from loss import masked_huber_loss


def calculate_target_values(model, target_model, state_transitions, discount_factor):
    states = []
    new_states = []

    for transition in state_transitions:
        states.append(transition.old_state)
        new_states.append(transition.new_state)

    states = np.array(states)
    new_states = np.array(new_states)

    q_values = LunarLanderAgent.get_multiple_q_values(model, states)
    q_values_target_model = LunarLanderAgent.get_multiple_q_values(target_model, states)

    q_values_new_state = LunarLanderAgent.get_multiple_q_values(model, new_states)
    q_values_new_state_target_model = LunarLanderAgent.get_multiple_q_values(target_model, new_states)

    targets = []
    for index, state_transition in enumerate(state_transitions):
        best_action = LunarLanderAgent.select_best_action(q_values_new_state[index])
        best_action_next_state_q_value = q_values_new_state_target_model[index][best_action]

        if state_transition.done:
            target_value = state_transition.reward
        else:
            target_value = state_transition.reward + discount_factor * best_action_next_state_q_value

        target_vector = [0, 0, 0, 0]
        target_vector[state_transition.action] = target_value
        targets.append(target_vector)

    return np.array(targets)


def train_model(model, states, targets):
    model.fit(states, targets, epochs=1, batch_size=len(targets), verbose=0)


def copy_model(model):
    backup_file = 'backup_'+str(uuid.uuid4())
    model.save(backup_file)
    new_model = load_model(backup_file, custom_objects={'masked_huber_loss': masked_huber_loss(0.0, 1.0)})
    shutil.rmtree(backup_file)
    return new_model
