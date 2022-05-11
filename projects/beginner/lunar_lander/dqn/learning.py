def calculate_target_values(model, target_model, state_transitions, discount_factor):
    states = []
    new_states = []
    for transition in state_transitions:
        states.append(transition.old_state)
        new_states.append(transition.new_state)

    new_states = np.array(new_states)

    q_values_new_state = get_multiple_q_values(model, new_states)
    q_values_new_state_target_model = get_multiple_q_values(target_model, new_states)

    targets = []
    for index, state_transition in enumerate(state_transitions):
        best_action = select_best_action(q_values_new_state[index])
        best_action_next_state_q_value = q_values_new_state_target_model[index][best_action]

        if state_transition.done:
            target_value = state_transition.reward
        else:
            target_value = state_transition.reward + discount_factor * best_action_next_state_q_value

        target_vector = [0] * outputs
        target_vector[state_transition.action] = target_value
        targets.append(target_vector)

    return np.array(targets)