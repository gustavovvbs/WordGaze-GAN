import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.activations import relu

# NORMALIZANDO AS COORDENADAS

# tensor = tf.random.normal((128, 3))

# coordinates = tensor[:, :2]

# min_values = tf.reduce_min(coordinates, axis=0)
# max_values = tf.reduce_max(coordinates, axis=0)

# normalized_coordinates = (coordinates - min_values) / (max_values - min_values) * 2 - 1

# normalized_tensor = tf.identity(tensor)
# normalized_tensor[:, :2].assign(normalized_coordinates)

class Encoder(Model):

    def __init__(self, input_dim=384, zdim=32):
        super().__init__()

        self.encoder = tf.keras.Sequential([
            layers.Dense(112, input_shape=(input_dim,), activation=layers.LeakyReLU(0.01)),
            layers.Dense(96, activation=layers.LeakyReLU(0.01)),
            layers.Dense(48, activation=layers.LeakyReLU(0.01)),
            layers.Dense(zdim, activation=layers.LeakyReLU(0.01))
        ])

        self.z_mean = layers.Dense(zdim)
        self.z_log_var = layers.Dense(zdim)

    def reparameterize(self, z_mu, z_log_var):
        epsilon = tf.random.normal(shape=z_log_var.shape)
        z = z_mu + epsilon * tf.exp(z_log_var/2)
        return z

    def call(self, inputs):
        x = self.encoder(inputs)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded, z_mean, z_log_var


if __name__ == "__main__":
    model = Encoder()
    print("Output dimension:", model(tf.random.normal((1, 384)))[0].shape)
