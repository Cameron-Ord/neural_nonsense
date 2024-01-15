import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
print()
print()
print()

user_inputs = [
    "Hello, how are you?"
]

responses = [
    "Great, thank you!"
]

max_sequence_length = 100

def vectorize_text():
    print("Vectorizing text..")
    print()
    vectorized_text = layers.TextVectorization(max_tokens=1000, output_mode='int')
    vectorized_text.adapt(user_inputs)
    inputs = vectorized_text(user_inputs)
    outputs = vectorized_text(responses)
    inputs = pad_sequences(inputs, maxlen=max_sequence_length, padding='post')
    outputs = pad_sequences(outputs, maxlen=max_sequence_length, padding='post')
    return inputs, outputs

def create_model():
    print("Creating model..")
    print()
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=1000, output_dim=10, input_length=max_sequence_length))
    model.add(layers.LSTM(units=64, return_sequences=True))
    model.add(layers.Dense(units=1000, activation='softmax'))
    return model

def summarize_model(model):
    model.summary()

def compile_model(model):
    print("Compiling")
    print()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def train_model(model, inputs, outputs):
    print("Training")
    print()
    model.fit(inputs, outputs, epochs=10, batch_size=32)

def save_model(model):
    print("Saving model")
    print()
    model.save('model1.keras')

rtr_inputs, rtr_outputs = vectorize_text()
rtr_model = create_model()
summarize_model(rtr_model)
compile_model(rtr_model)
train_model(rtr_model, rtr_inputs, rtr_outputs)
save_model(rtr_model)

