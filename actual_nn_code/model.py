import tensorflow as tf
from tensorflow import keras
from keras import layers

##Inheriting from keras.Model
##The instance of the class will now contain the model

class TextModel(keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        print("Creating model from constructor")
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        
        ## Inputs is a tensor object
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x 




