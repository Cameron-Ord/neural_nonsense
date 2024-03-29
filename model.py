import tensorflow as tf
from tensorflow import keras
from keras import layers

##Inheriting from keras.Model
##The instance of the class will now contain the model

class TextModel(keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, **kwargs):
        super(TextModel, self).__init__(**kwargs)
        print("Creating model from constructor")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(rnn_units, return_sequences=True, return_state=True)
        self.dense = layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        ## Inputs is a tensor object
        x = inputs
        x = self.embedding(x, training=training)
        x, states_h, states_c = self.lstm(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, [states_h, states_c]
        else:
            return x 
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'rnn_units': self.rnn_units,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)