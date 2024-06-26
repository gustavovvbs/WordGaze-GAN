import tensorflow as tf 
from tensorflow.keras import layers, Model 
from tensorflow.layers import SpectralNormalization
import pandas as pd 
import numpy as np
import os 
import scipy.special as sp

###<--------VARIATIONAL ENCODER------->###

class VariationalEncoder(tf.keras.Model):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(192, activation = tf.keras.layers.LeakyReLU()),
            layers.Dense(96, activation = tf.keras.layers.LeakyReLU()),
            layers.Dense(48, activation = tf.keras.layers.LeakyReLU()),
            layers.Dense(32, activation = tf.keras.layers.LeakyReLU())
        ])

        self.mu = layers.Dense(32)
        self.log_var = layers.Dense(32)

    def call(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(shape = mu.shape)
        return eps * tf.exp(log_var * 0.5) + mu

    def L_lat(self, z, z_generated):
        """
        Calculate the latent encoding loss L_lat

        :param z: The original latent variable
        :param z_generated: The re-encoded latent variable
        """

        # Calculate the L1 norm of the difference between the original latent variable and the re-encoded latent variable   
        
        return tf.reduce_mean(tf.abs(z - z_generated))

    def L_KLD(self, real_data_flat):
        """"
        Calculate the Kullback-Leibler divergence loss L_KLD

        :param mu: The mean vector produced by the variational encoder
        :param log_var: The log variance vector produced by the variational encoder
        :return: The scalar value of the KL divergence loss
        """

        real_data = tf.reshape(real_data_flat, [real_data_flat.shape[0], 128, 3])
        mu, log_var = self.call(real_data)
        
        return sp.kl_div(mu, log_var)

    
###<------------DISCRIMINATOR------------>###

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        #spectral normalization layers 
        self.discriminator = tf.keras.Sequential([
            SpectralNormalization(tf.keras.layers.Dense(192, activation = tf.keras.layers.LeakyReLU())),
            SpectralNormalization(tf.keras.layers.Dense(96, activation = tf.keras.layers.LeakyReLU())),
            SpectralNormalization(tf.keras.layers.Dense(48, activation = tf.keras.layers.LeakyReLU())),
            SpectralNormalization(tf.keras.layers.Dense(24, activation = tf.keras.layers.LeakyReLU())),
            SpectralNormalization(tf.keras.layers.Dense(1, activation = tf.keras.layers.LeakyReLU()))
        ])

    def call(self, x):
        return self.discriminator(x)

    #disc-loss
    def disc_loss(self, fake_data_flat, real_data_flat):
        fake_output = self.discriminator(fake_data_flar, training = True)
        real_output = self.discriminator(real_data_flat, training = True)

        #wasserstein distance(minimize the distance between hopes)
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    #feature extraction 
    def extract_features(self, x):
        features = []
        #all layers except the last one
        for layer in self.discriminator.layers[:-1]:
            x = layer(x)
            features.append(x)
        
        return features

###<------------GENERATOR------------>###

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.bilstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences = True, activation = 'tanh'), input_shape = (35, 32))
        self.bilstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences = True, activation = 'tanh'))
        self.bilstm3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences = True, activation = 'tanh'))
        self.bilstm4 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences = True, activation = 'tanh'))
        
        self.dense = tf.keras.layers.Dense(3, activation = 'tanh')

    def call(self, x):
        x = self.bilstm1(x)
        x = self.bilstm2(x)
        x = self.bilstm3(x)
        x = self.bilstm4(x)

        return self.dense(x)

    #generator loss 
    def gen_loss(self, discriminator, encoder, fake_data_flat, real_data_flat, z, z_generated):
        fake_output = discriminator(fake_data_flat, training = True)
        real_output = discriminator(real_data_flat, training = True)

        loss_G = -discriminator.disc_loss(fake_data_flat, real_data_flat)\
                + lambda_feat + self.L_feat(discriminator, fake_data_flat, real_data_flat)\
                + lambda_lat * encoder.L_lat(z, z_generated)\
                + lambda_KLD * encoder.L_KLD(real_data_flat)\
                + lambda_rec * self.L_rec(fake_output, real_output)

        return loss_G

    #compute feature matching loss 
    def L_feat(self, discriminator, fake_output, real_output):
        loss = 0
        fake_features = discriminator.extract_features(fake_output)
        real_features = discriminator.extract_features(real_output)

        for fake_feature, real_feature in zip(fake_features, real_features):
            loss += tf.reduce_mean(tf.abs(fake_feature - real_feature))
        
        return loss

    #compute latent reconstruction loss
    def L_rec(self, fake_output, real_output):
        return tf.reduce_mean(tf.abs(fake_output - real_output))
