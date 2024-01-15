import numpy as np
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import lists
model = keras.models.load_model('dagoth_textbot_0.1.keras')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
responses = lists.get_resp()

def vectorize_input(user_input):
    print(user_input)
    vt_input = layers.TextVectorization(max_tokens=1000, output_mode='int', input_shape=(None,))
    tensor_data = tf.constant(user_input, dtype=tf.string)
    vt_input.adapt(tensor_data)
    input = vt_input(tensor_data) 
    input = pad_sequences(input, maxlen=1000, padding='post')
    return vt_input, input

def vectorize_output():
    vt_output = layers.TextVectorization(max_tokens=1000, output_mode='int', input_shape=(None,))
    tensor_data  = tf.constant(responses, dtype=tf.string)
    vt_output.adapt(tensor_data)
    output = vt_output(tensor_data) 
    output = pad_sequences(output, maxlen=1000, padding='post')
    return vt_output, output

def predict(input, vt_out):
    output = model.predict(input)
    indices = np.argmax(output, axis=-1)
    vocab = vt_out.get_vocabulary()
    
    resp_arr = []
    for sequence in indices:
        concat_resp = ''
        try:
            predicted_text = [vocab[seq] for seq in sequence]
        
            for token in predicted_text:
                if len(token) > 0:
                    concat_resp += token + " "
            if len(concat_resp) > 0:
                resp_arr.append(concat_resp)
        except Exception as e:
            print(e)

    print(resp_arr)

running = True
while running:
 try:
    print("Enter your query...")
    user_input = str(input())
    input_list = []
    input_list.append(user_input)
    vt_in, input_seq = vectorize_input(input_list)
    vt_out, output_seq = vectorize_output()
    predict(input_seq, vt_out)
 except Exception as e:
        print("Something went wrong: ", e)
 else:
    pass
