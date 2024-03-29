import tensorflow as tf
from tensorflow import keras
import numpy as np
import datasets
from model import TextModel
url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'

class Model_Init():
    def __init__(self):
        self.path_to_file = keras.utils.get_file('shakespeare.txt', url)
        self.enter_nn_name()
        vocab, text = self.read_and_decode()
        self.create_chars_from_vocab(vocab)
        self.create_ids_from_chars(vocab)
        self.create_chars_from_ids()
        self.create_targets(text)
    
    def enter_nn_name(self): 
        print("Enter name for NN: ")
        self.name = str(input())

    def create_chars_from_vocab(self, vocab):
        self.chars = tf.strings.unicode_split(vocab, input_encoding="UTF-8")

    def text_from_ids(self, ids):
        return tf.strings.reduce_join(self.chars_from_ids(ids), axis=-1)

    def create_chars_from_ids(self):
        self.chars_from_ids = keras.layers.StringLookup(
            vocabulary=self.ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

    def split_input_target(self, sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        print("Sequence: \n",sequence, "\n")
        print("Input: \n",input_text, "\n")
        print("Target: \n",target_text, "\n")
        return input_text, target_text

    def create_targets(self, text):
        print("Creating targets..\n\n", text)
        seq_length = 250
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
        print(f'Vocab: \n {vocab}')
        return vocab, text

    def create_batches(self, dataset):
        BATCH_SIZE = 64
        BUFFER_SIZE = 10000
        train_dataset = (
            dataset
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        self.create_model(train_dataset)

    def create_model(self, dataset):
        print("Creating a new model..")
        vocab_size = len(self.ids_from_chars.get_vocabulary())
        print("VOCAB SIZE: \n", vocab_size)
        embedding_dim = 256
        rnn_units = 1024
        model_instance = TextModel(vocab_size, embedding_dim, rnn_units)         
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        model_instance.compile(optimizer='adam', loss=loss, metrics=['accuracy', 'sparse_categorical_accuracy'])
        self.train_model(dataset, model_instance)

    def train_model(self, data, model):
        EPOCHS = 10
        history = model.fit(data, epochs=EPOCHS)
        model.summary()
        model.save(self.name + '.keras')


if __name__=="__main__":
    init_instance = Model_Init()
