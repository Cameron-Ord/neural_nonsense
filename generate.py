import tensorflow as tf
from tensorflow import strings
from tensorflow import keras
from keras import models
import numpy as np

class Model_Start():
    def __init__(self):
        self.path_to_file = 'dagoth.txt'
        vocab = self.read_and_decode()
        self.create_ids_from_chars(vocab)
        self.create_chars_from_ids()
        self.load_nn()
        self.run_nn()

    def create_chars_from_ids(self):
        self.chars_from_ids = keras.layers.StringLookup(
            vocabulary=self.ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

    def text_from_ids(self, ids):
        return tf.strings.reduce_join(self.chars_from_ids(ids), axis=-1)

    def create_ids_from_chars(self, vocab):
        self.ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None) 

    def read_and_decode(self):
        with open(self.path_to_file, 'r') as file:
            text = file.read()
        print(f'Length of text: {len(text)} characters', "\n")
        vocab = sorted(set(text))
        print(f'{len(vocab)} unique characters', "\n")
        return vocab

    def load_nn(self):
        print("Enter model name: ")
        model_name = str(input())
        self.model = models.load_model(model_name + '.keras', compile=False, custom_objects=None)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def run_nn(self):
        running = True
        while running:
            print("Your Message: ")
            seed_text = str(input())
            text = self.generate_text(seed_text)

    def generate_text(self, seed_text):
        input_batch = self.ids_from_chars(strings.unicode_split(seed_text, 'UTF-8'))
        input_batch = tf.expand_dims(input_batch, 0)
        predictions = self.model(input_batch)
        print("Predictions Shape:", predictions.shape)
        sampled_indices = tf.random.categorical(predictions[0], num_samples=1)
        sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
        print("Indices: ", sampled_indices)
        print("Next char predictions: \n", self.text_from_ids(sampled_indices).numpy())
            

if __name__=="__main__":
    init_instance = Model_Start()
