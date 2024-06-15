import tensorflow as tf
from tensorflow.keras import layers, Model


class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        
        #repete os encodings na dimensao e concatena
        self.repeat_vector = layers.RepeatVector(128)
        self.reshape_coding = layers.Reshape((128, 32))
        self.concatenate = layers.Concatenate(axis=-1)
        

        self.lstm1 = layers.Bidirectional(layers.LSTM(32, return_sequences=True))
        self.lstm2 = layers.Bidirectional(layers.LSTM(32, return_sequences=True))
        self.lstm3 = layers.Bidirectional(layers.LSTM(32, return_sequences=True))
        
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(128 * 3, activation='tanh')
        self.reshape_output = layers.Reshape((128, 3))

    def call(self, inputs):
        coding_input, vector_input = inputs
        
        coding_repeated = self.repeat_vector(coding_input)
        coding_reshaped = self.reshape_coding(coding_repeated)
        
        concatenated = self.concatenate([coding_reshaped, vector_input])
        
        x = self.lstm1(concatenated)
        x = self.lstm2(x)
        x = self.lstm3(x)
        
        x = self.flatten(x)
        output = self.dense(x)
        output = self.reshape_output(output)
        
        return output


if __name__ == '__main__':
    generator = Generator()
    generator.build(input_shape=[(None, 32), (None, 128, 3)])
    print(generator.summary())
    #test output shape
    print(generator([tf.random.normal((1, 32)), tf.random.normal((1, 128, 3))]).shape)

