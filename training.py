import numpy as np
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import lists
print()
print()
print()

user_inputs = lists.get_inp()
dagoth_ur_responses = lists.get_resp()

max_sequence_length = 150

def vectorize_text():
    print("Vectorizing text..")
    print()
    vectorized_text = layers.TextVectorization(max_tokens=300, output_mode='int', input_shape=(None,))
    vectorized_text.adapt(user_inputs + dagoth_ur_responses)
    inputs = vectorized_text(user_inputs)
    outputs = vectorized_text(dagoth_ur_responses)
    inputs = pad_sequences(inputs, maxlen=max_sequence_length, padding='post')
    outputs = pad_sequences(outputs, maxlen=max_sequence_length, padding='post')
    return inputs, outputs, vectorized_text

def create_model(vt, inputs, outputs):
    print("Creating model..")
    print()
    vocab_size = len(vt.get_vocabulary())

    
    try:
        loaded_model = keras.models.load_model('dagoth_textbot_0.1.keras')
        summarize_model(loaded_model)
        compiled_model = compile_model(loaded_model)
        train_model(inputs, outputs, compiled_model)
        
    except Exception as e:
        print('model does not exist, creating a new model')
        new_model = create_new(vocab_size)
        summarize_model(new_model)
        compiled_model = compile_model(new_model)
        train_model(inputs, outputs, compiled_model)

def create_new(vocab_size):
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sequence_length))
    model.add(layers.LSTM(units=64, return_sequences=True))
    model.add(layers.Dense(units=vocab_size, activation='softmax'))
    save_model(model)
    return model

def summarize_model(l_mdl):
    l_mdl.summary()

def compile_model(sent_model):
    print("Compiling")
    print()
    sent_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return sent_model

i = 0

def train_model(inputs, outputs, compiled_model):
    global i
    print("Training")
    print()
    
    compiled_model.fit(inputs, outputs, epochs=100, batch_size=64)
    print("TRAINING INCREMENTER: ", i)
    if i < 20:
        i+=1
        save_model(compiled_model)
        train_model(inputs, outputs, compiled_model)   
    
   

def save_model(model):
    print("Saving model")
    print()
    model.save('dagoth_textbot_0.1.keras')

def predict_response(txt_layer, inputs):
    loaded_model = keras.models.load_model('dagoth_textbot_0.1.keras')
    compiled_model = compile_model(loaded_model)
    intr_outputs = compiled_model.predict(inputs)
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

if __name__ == "__main__":
    rtr_inputs, rtr_outputs, vt = vectorize_text()
    create_model(vt, rtr_inputs, rtr_outputs)
    predict_response(vt, rtr_inputs)

