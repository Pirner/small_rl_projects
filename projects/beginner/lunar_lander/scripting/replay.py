import random


class ReplayBuffer:
    current_index = 0

    def __init__(self, size=10000):
        self.size = size
        self.transitions = []

    def add(self, transition):
        if len(self.transitions) < self.size:
            self.transitions.append(transition)
        else:
            self.transitions[self.current_index] = transition
            self.__increment_current_index()

    def length(self):
        return len(self.transitions)

    def get_batch(self, batch_size):
        return random.sample(self.transitions, batch_size)

    def __increment_current_index(self):
        self.current_index += 1
        if self.current_index >= self.size - 1:
            self.current_index = 0
