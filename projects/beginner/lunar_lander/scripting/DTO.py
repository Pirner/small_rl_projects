class StateTransition:
    def __init__(self, old_state, action, reward, new_state, done):
        self.old_state = old_state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.done = done
