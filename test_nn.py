import numpy as np
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
print()
print()
print()



user_inputs = [
    "Greetings, Dagoth Ur.",
    "What is the purpose of the Sixth House?",
    "Tell me about the dreams you offer.",
    "Are you friend or foe?",
    "Explain the power of the Heart of Lorkhan.",
    "How can I join the Sixth House?",
    "What do you think of the Nerevarine?",
    "What is your vision for Morrowind's future?",
    "Speak, Lord Dagoth."
]


dagoth_ur_responses = [
    "Ah, mortal. Greetings. What do you seek?",
    "The purpose of the Sixth House is the restoration of Morrowind's glory.",
    "Dreams of power, dreams of conquest. All within the embrace of the Heart.",
    "Friend or foe matters little. The Sixth House will embrace all in time.",
    "The Heart grants power, unimaginable power. It is the key to ascension.",
    "To join the Sixth House is to embrace the power of the Heart. Are you ready?",
    "The Nerevarine? A pawn of fate, but ultimately inconsequential.",
    "The future of Morrowind lies in the shadow of the Sixth House.",
    "Speak, mortal. Dagoth Ur listens."
]


max_sequence_length = 100

def vectorize_text():
    print("Vectorizing text..")
    print()
    vectorized_text = layers.TextVectorization(max_tokens=1500, output_mode='int')
    vectorized_text.adapt(user_inputs + dagoth_ur_responses)
    inputs = vectorized_text(user_inputs)
    outputs = vectorized_text(dagoth_ur_responses)
    inputs = pad_sequences(inputs, maxlen=max_sequence_length, padding='post')
    outputs = pad_sequences(outputs, maxlen=max_sequence_length, padding='post')
    return inputs, outputs, vectorized_text

def create_model():
    print("Creating model..")
    print()
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=1500, output_dim=125, input_length=max_sequence_length))
    model.add(layers.LSTM(units=64, return_sequences=True))
    model.add(layers.Dense(units=1500, activation='softmax'))
    return model

def summarize_model(model):
    model.summary()

def compile_model(model):
    print("Compiling")
    print()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

i = 0

def train_model(model, inputs, outputs):
    global i
    print("Training")
    print()
    model.fit(inputs, outputs, epochs=200, batch_size=64)
    print("TRAINING INCREMENTER: ", i)
    if i < 10:
        i+=1
        save_model(model)
        train_model(model, inputs, outputs)   

    return model

def save_model(model):
    print("Saving model")
    print()
    model.save('dagoth_textbot_0.1.keras')

def predict_response(model, txt_layer, inputs):
    intr_outputs = model.predict(inputs)
    predicted_indices = np.argmax(intr_outputs, axis=-1)
    vocab = txt_layer.get_vocabulary()
    resp_arr = []
    for sequence in predicted_indices:
        concat_resp = ''
        predicted = [vocab[seq] for seq in sequence]
        for token in predicted:
            if len(token) > 0:
                concat_resp += token + " "
        if len(concat_resp) > 0:
            resp_arr.append(concat_resp)
    print(resp_arr)

rtr_inputs, rtr_outputs, txt_layer = vectorize_text()
rtr_model = create_model()
summarize_model(rtr_model)
compile_model(rtr_model)
trained_model = train_model(rtr_model, rtr_inputs, rtr_outputs)
predict_response(trained_model, txt_layer, rtr_inputs)


