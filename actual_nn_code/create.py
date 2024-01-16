import tensorflow as tf
import numpy as np
import os, copy
import time
from model import TextModel

url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'

class Model_Init():
    def __init__(self):
        
        self.path_to_file = tf.keras.utils.get_file('shakespeare.txt', url)
        vocab, text = self.read_and_decode()
        ids = self.create_numeric_ids(vocab)
        chars = self.chars_from_ids(ids)
        self.create_targets(text, chars)
    
    def text_from_ids(self, ids):
        return tf.strings.reduce_join(self.chars_from_ids(ids), axis=-1)

    def chars_from_ids(self, ids):
        self.chars_from_ids = tf.keras.layers.StringLookup(
            vocabulary=self.ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
        chars = self.chars_from_ids(ids)
        return chars

    def split_input_target(self, sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    def create_targets(self, text, chars):
        seq_length = 100
        all_ids = self.ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
        sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
        dataset = sequences.map(self.split_input_target)
        self.create_batches(dataset)

    def create_numeric_ids(self, vocab):
        chars = tf.strings.unicode_split(vocab, input_encoding="UTF-8")
        self.ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
        ids = self.ids_from_chars(chars)
        return ids

    def read_and_decode(self):
        text = open(self.path_to_file, 'rb').read().decode(encoding='utf-8')
        print(f'Length of text: {len(text)} characters')
        vocab = sorted(set(text))
        print(f'{len(vocab)} unique characters')
        return vocab, text

    def create_batches(self, dataset):
        BATCH_SIZE = 64
        BUFFER_SIZE = 10000

        train_dataset = (
            dataset
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))
        print(train_dataset)
        self.create_model(train_dataset)

    def create_model(self, dataset):
        vocab_size = len(self.ids_from_chars.get_vocabulary())
        embedding_dim = 256
        rnn_units = 1024 
        model_instance = TextModel(vocab_size, embedding_dim, rnn_units)      
        for input_batch, target_batch in dataset.take(1):
            batch_predictions = model_instance(input_batch)
            print(batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
        model_instance.summary()
    

if __name__=="__main__":
    init_instance = Model_Init()
