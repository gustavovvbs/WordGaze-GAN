#forward passa through the network
import numpy as np
import math
import random
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from encoder import *
from generator import *
from discriminator import *
from prototype import *


realdata = tf.random.normal([1, 128, 3])
latent_dim = 32
generator = Generator()
discriminator = Discriminator()
variational = VariationalAutoencoder(latent_dim)

#forward pass through the network
z, mu, log_var = variational(realdata)
generated = generator([z, get_batch_prototypes(['nigga'])])
#plot the points after passing through generator
plt.plot(generated[0, :, 0], -generated[0, :, 1], 'bo', linestyle='-')
plt.show()
discriminator_output = discriminator(generated) 
print("Discriminator output:", discriminator_output)


