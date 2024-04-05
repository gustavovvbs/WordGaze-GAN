import tensorflow as tf
from tensorflow.keras import layers, Model

class Discriminator(Model):
    def __init__(self, input_size=384):
        super().__init__()

        self.discriminator = tf.keras.Sequential([
            layers.Dense(192, input_shape=(input_size, ), activation = layers.LeakyReLU(0.01)),
            layers.Dense(96, activation = layers.LeakyReLU(0.01)),
            layers.Dense(48, activation = layers.LeakyReLU(0.01)),
            layers.Dense(24, activation = layers.LeakyReLU(0.01)),
            layers.Dense(1, activation = layers.LeakyReLU(0.01)),
        ])

    def call(self, x):
        x = self.discriminator(x)
        return x

if __name__ == '__main__':
    model = Discriminator()
    print("Output dimension:", model(tf.random.normal((1, 384)))[0].shape)