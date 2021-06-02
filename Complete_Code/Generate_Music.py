

import os
import json
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, Embedding


d_dir = "../Data2/"
d_file = "d_tunes.txt"
charIndex_json = "char_to_index.json"
m_q_dir = '../Data2/Model_Weights/'
b_size = 16
s_len = 64


def making_the_model(u_char):
    model = Sequential()
    
    model.add(Embedding(input_dim = u_char, output_dim = 512, batch_input_shape = (1, 1))) 
  
    model.add(LSTM(256, return_sequences = True, stateful = True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(256, return_sequences = True, stateful = True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(256, stateful = True)) 
    model.add(Dropout(0.2))
    
    model.add((Dense(u_char)))
    model.add(Activation("softmax"))
    
    return model



def generating_the_sequence(epoch_num, initial_index, s_len):
    with open(os.path.join(d_dir, charIndex_json)) as f:
        char_to_index = json.load(f)
    index_to_char = {i:ch for ch, i in char_to_index.items()}
    u_char = len(index_to_char)
    
    model = making_the_model(u_char)
    model.load_weights(m_q_dir + "Weights_{}.h5".format(epoch_num))
     
    sequence_index = [initial_index]
    
    for _ in range(s_len):
        batch = np.zeros((1, 1))
        batch[0, 0] = sequence_index[-1]
        
        predicted_probs = model.predict_on_batch(batch).ravel()
        sample = np.random.choice(range(u_char), size = 1, p = predicted_probs)
        
        sequence_index.append(sample[0])
    
    seq = ''.join(index_to_char[c] for c in sequence_index)
    
    cnt = 0
    for i in seq:
        cnt += 1
        if i == "\n":
            break
    seq1 = seq[cnt:]
    
    
    cnt = 0
    for i in seq1:
        cnt += 1
        if i == "\n" and seq1[cnt] == "\n":
            break
    seq2 = seq1[:cnt]
    
    return seq2


ep = int(input("1. Enter epoch number(10, 20, 30, ..., 90).: "))
ar = int(input("\n2. Enter any number between 0 to 86 which will be given as initial charcter to model for generating sequence: "))
ln = int(input("\n3. Enter the length of music sequence between 300-600: "))

music = generating_the_sequence(ep, ar, ln)

print("\nTHE GENERATED MUSIC SEQUENCE IS AS: \n")

print(music)
