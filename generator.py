import tensorflow as tf
from tensorflow.keras import layers, Model


class Generator(Model):
    def __init__(self, input_size):
        super().__init__()

        self.lstm1 = layers.Bidirectional(layers.LSTM(32, return_sequences=True))
        self.lstm2 = layers.Bidirectional(layers.LSTM(32, return_sequences=True))
        self.lstm3 = layers.Bidirectional(layers.LSTM(32, return_sequences=True))
        self.fc1 = layers.Dense(3)

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)  
        x = self.lstm1(x)
        print(x.shape)
        x = self.lstm2(x)
        print(x.shape)
        x = self.lstm3(x)
        print(x.shape)
        x = tf.keras.layers.Flatten()(x)  
        print(x.shape)
        output = self.fc1(x)
        output = tf.math.tanh(output)
        return output

if __name__ == '__main__':
    model = Generator(input_size=35)
    print('dimension of the output', model(tf.random.normal((128,35))).shape)
