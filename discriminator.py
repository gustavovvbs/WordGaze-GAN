import tensorflow as tf
from tensorflow.keras import layers, models

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten1 = layers.Flatten()
        self.dense1 = layers.Dense(192, activation='linear')
        self.leaky_relu1 = layers.LeakyReLU()
        self.dense2 = layers.Dense(96, activation='linear')
        self.leaky_relu2 = layers.LeakyReLU()
        self.dense3 = layers.Dense(48, activation='linear')
        self.leaky_relu3 = layers.LeakyReLU()
        self.dense4 = layers.Dense(24, activation='linear')
        self.leaky_relu4 = layers.LeakyReLU()
        self.dense5 = layers.Dense(1, activation='linear')
        self.leaky_relu5 = layers.LeakyReLU()

    def call(self, inputs):
        x = self.flatten1(inputs)
        x = self.dense1(x)
        x = self.leaky_relu1(x)
        x = self.dense2(x)
        x = self.leaky_relu2(x)
        x = self.dense3(x)
        x = self.leaky_relu3(x)
        x = self.dense4(x)
        x = self.leaky_relu4(x)
        x = self.dense5(x)
        x = self.leaky_relu5(x)
        return x

# Example usage
discriminator = Discriminator()

# Dummy input representing a user-drawn/simulated gesture
input_gesture = tf.random.normal([128, 3])
d_output = discriminator(input_gesture)
print("Discriminator output:", d_output)
