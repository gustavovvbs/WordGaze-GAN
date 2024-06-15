import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

class VariationalEncoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VariationalEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(192, activation='linear')
        self.leaky_relu1 = layers.LeakyReLU()
        self.dense2 = layers.Dense(96, activation='linear')
        self.leaky_relu2 = layers.LeakyReLU()
        self.dense3 = layers.Dense(48, activation='linear')
        self.leaky_relu3 = layers.LeakyReLU()
        self.dense_mu = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.leaky_relu1(x)
        x = self.dense2(x)
        x = self.leaky_relu2(x)
        x = self.dense3(x)
        x = self.leaky_relu3(x)
        mu = self.dense_mu(x)
        log_var = self.dense_log_var(x)
        return mu, log_var

class Reparameterize(layers.Layer):
    def call(self, inputs):
        mu, log_var = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = mu + tf.exp(0.5 * log_var) * epsilon
        return z

class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dim)
        self.reparameterize = Reparameterize()
        # Add decoder if necessary

    def call(self, inputs):
        mu, log_var = self.encoder(inputs)
        z = self.reparameterize((mu, log_var))
        # Decode if necessary
        return z, mu, log_var

# Example usage
latent_dim = 32
vae = VariationalAutoencoder(latent_dim)

# Dummy input representing a user-drawn gesture
input_gesture = tf.random.normal([1, 128, 3])
z, mu, log_var = vae(input_gesture)
print("Encoded latent vector:", z.shape)  # Should be (1, 32)
