import tensorflow as tf
import numpy as np
import os, copy
import time
from model import TextModel
url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'

class Model_Init():
    def __init__(self):
        print()
        print()
        self.path_to_file = tf.keras.utils.get_file('shakespeare.txt', url)
        vocab, text = self.read_and_decode()
        self.create_chars_from_vocab(vocab)
        self.create_ids_from_chars(vocab)
        self.create_chars_from_ids()
        self.create_targets(text)
    
    def create_chars_from_vocab(self, vocab):
        self.chars = tf.strings.unicode_split(vocab, input_encoding="UTF-8")

    def text_from_ids(self, ids):
        return tf.strings.reduce_join(self.chars_from_ids(ids), axis=-1)

    def create_chars_from_ids(self):
        self.chars_from_ids = tf.keras.layers.StringLookup(
            vocabulary=self.ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

    def split_input_target(self, sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        print("    ",sequence, "\n")
        print("    ",input_text, "\n")
        print("    ",target_text, "\n")
        return input_text, target_text

    def create_targets(self, text):
        seq_length = 100
        all_ids = self.ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
        sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
        dataset = sequences.map(self.split_input_target)
        self.create_batches(dataset)

    def create_ids_from_chars(self, vocab):
        self.ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None) 

    def read_and_decode(self):
        text = open(self.path_to_file, 'rb').read().decode(encoding='utf-8')
        print(f'Length of text: {len(text)} characters', "\n")
        vocab = sorted(set(text))
        print(f'{len(vocab)} unique characters', "\n")
        return vocab, text

    def create_batches(self, dataset):
        BATCH_SIZE = 64
        BUFFER_SIZE = 10000
        print("DATASET PRIOR: \n", dataset ,"\n")
        train_dataset = (
            dataset
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        print("DATASET AFTER: \n", train_dataset, "\n")
        print("load or create new model? l/n", "\n")
        
        chosen = False
        while(chosen != True):
            input_str = str(input())
            if(input_str == "l"):
                print()
                self.load_model(train_dataset)
                chosen = True
            elif(input_str == "n"):
                print()
                self.create_model(train_dataset)
                chosen = True
            else:
                print("Enter either l for load or n for new", "\n")

    def test_run(self, dataset, model_instance):
        for input_batch, target_batch in dataset.take(5):
            batch_predictions = model_instance(input_batch)
            sampled_indices = tf.random.categorical(batch_predictions[0], num_samples=1)
            sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
            print("----------------------------", "\n")
            print("Input: \n", self.text_from_ids(input_batch[0]).numpy(), "\n")
            print("Next Char predictions: \n", self.text_from_ids(sampled_indices).numpy())
            print("----------------------------", "\n")
            loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
            example_batch_mean_loss = loss(target_batch, batch_predictions)
            print("----------------------------", "\n") 
            print("Prediction shape: ", batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)", "\n")
            print("Mean loss:        ", example_batch_mean_loss, "\n")
            exponential_loss = tf.exp(example_batch_mean_loss).numpy()
            print("Exponetial loss: ", exponential_loss, "\n")
            print("----------------------------", "\n")

    def load_model(self, dataset):
        print("Loading existing model..")
        model_instance = tf.keras.models.load_model('horatio_bot.keras')
        self.test_run(dataset, model_instance)
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        model_instance.compile(optimizer='adam', loss=loss)
        self.train_model(dataset, model_instance)
        

    def create_model(self, dataset):
        print("Creating a new model..")
        vocab_size = len(self.ids_from_chars.get_vocabulary())
        embedding_dim = 256
        rnn_units = 1024 
        model_instance = TextModel(vocab_size, embedding_dim, rnn_units)         
        self.test_run(dataset, model_instance)
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        model_instance.compile(optimizer='adam', loss=loss)
        self.train_model(dataset, model_instance)

    def train_model(self, data, model):
        EPOCHS = 3
        history = model.fit(data, epochs=EPOCHS)
        model.save("horatio_bot.keras")


if __name__=="__main__":
    init_instance = Model_Init()
