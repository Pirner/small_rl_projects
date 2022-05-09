import tensorflow as tf


class Critic(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(2048, activation='relu')
        self.d2 = tf.keras.layers.Dense(1536, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        v = self.v(x)
        return v


class Actor(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self, action_space):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(2048, activation='relu')
        self.d2 = tf.keras.layers.Dense(1536, activation='relu')
        self.a = tf.keras.layers.Dense(action_space, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        a = self.a(x)
        return a
