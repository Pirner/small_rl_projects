import random


class StateTransition():
    def __init__(self, old_state, action, reward, new_state, done):
        self.old_state = old_state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.done = done


class ReplayBuffer:
    def __init__(self, size=10000):
        self._size = size
        self._transitions = []
        self._current_index = 0

    def add(self, transition):
        if len(self._transitions) < self._size:
            self._transitions.append(transition)
        else:
            self._transitions[self._current_index] = transition
            self.__increment_current_index()

    def length(self):
        return len(self._transitions)

    def get_batch(self, batch_size):
        return random.sample(self._transitions, batch_size)

    def __increment_current_index(self):
        self.current_index += 1
        if self.current_index >= self._size - 1:
            self.current_index = 0
