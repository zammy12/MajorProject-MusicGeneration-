


import os
import json
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding

d_dir = "../Data2/"
d_file = "d_tunes.txt"
charIndex_json = "char_to_index.json"
m_w_dir = '../Data2/Model_Weights/'
b_size = 16
s_len = 64


def read_all(all_chars, u_char):
    len = all_chars.shape[0]
    b_char = int(len / b_size)
    
    for start in range(0, b_char - s_len, 64): 
        X = np.zeros((b_size, s_len))   
        Y = np.zeros((b_size, s_len, u_char))  
        for b_ind in range(0, 16):   
            for i in range(0, 64): 
                X[b_ind, i] = all_chars[b_ind * b_char + start + i]
                Y[b_ind, i, all_chars[b_ind * b_char + start + i + 1]] = 1 
        yield X, Y
        


def solve(b_size, s_len, u_char):
    model = Sequential()
    
    model.add(Embedding(input_dim = u_char, output_dim = 512, batch_input_shape = (b_size, s_len), name = "embd_1")) 
    
    model.add(LSTM(256, return_sequences = True, stateful = True, name = "lstm_first"))
    model.add(Dropout(0.2, name = "drp_1"))
    
    model.add(LSTM(256, return_sequences = True, stateful = True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(256, return_sequences = True, stateful = True))
    model.add(Dropout(0.2))
    
    model.add(TimeDistributed(Dense(u_char)))
    model.add(Activation("softmax"))
    
    model.load_weights("../Data/Model_Weights/Weights_80.h5", by_name = True)
    
    return model
def training_model(data, epochs = 90):
    char_to_index = {ch: i for (i, ch) in enumerate(sorted(list(set(data))))}
    print("Number of unique characters in our whole tunes database = {}".format(len(char_to_index))) #87
    
    with open(os.path.join(d_dir, charIndex_json), mode = "w") as f:
        json.dump(char_to_index, f)
        
    index_to_char = {i: ch for (ch, i) in char_to_index.items()}
    u_char = len(char_to_index)
    
    model = solve(b_size, s_len, u_char)
    model.summary()
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    
    all_characters = np.asarray([char_to_index[c] for c in data], dtype = np.int32)
    print("Total number of characters = "+str(all_characters.shape[0])) 
    
    epoch_number, loss, accuracy = [], [], []
    
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch+1, epochs))
        final_epoch_loss, final_epoch_accuracy = 0, 0
        epoch_number.append(epoch+1)
        
        for i, (x, y) in enumerate(read_all(all_characters, u_char)):
            final_epoch_loss, final_epoch_accuracy = model.train_on_batch(x, y) 
            print("Batch: {}, Loss: {}, Accuracy: {}".format(i+1, final_epoch_loss, final_epoch_accuracy))
        loss.append(final_epoch_loss)
        accuracy.append(final_epoch_accuracy)
        if (epoch + 1) % 10 == 0:
            if not os.path.exists(m_w_dir):
                os.makedirs(m_w_dir)
            model.save_weights(os.path.join(m_w_dir, "Weights_{}.h5".format(epoch+1)))
            print('Saved Weights at epoch {} to file Weights_{}.h5'.format(epoch+1, epoch+1))
   
    log_frame = pd.DataFrame(columns = ["Epoch",   "Loss",   "Accuracy"])
    log_frame["Epoch"] = epoch_number
    log_frame["Loss"] = loss
    log_frame["Accuracy"] = accuracy
    log_frame.to_csv("../Data2/log.csv", index = False)
    
file = open(os.path.join(d_dir, d_file), mode = 'r')
data = file.read()
file.close()
if __name__ == "__main__":
    training_model(data)
    
    
    
log = pd.read_csv(os.path.join(d_dir, "log.csv"))
log
