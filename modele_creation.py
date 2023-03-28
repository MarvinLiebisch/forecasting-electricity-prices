from energy_price_predictions.ml_logic.data_import import import_merged_data
from energy_price_predictions.ml_logic.preprocessing import normalize_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU,Dense
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import random
from tensorflow.keras.optimizers import Nadam
from keras.models import load_model

df=import_merged_data().drop(columns=['price_actual'])
df['price_day_ahead_shifted'] = df['price_day_ahead'].shift(24)
df = df.dropna()
X, y=normalize_dataset(df)
X=pd.DataFrame(X)
y=pd.DataFrame(y)

def sequence_data(X, y,
                  n_observation_X, n_observation_y,
                  n_sequence_train,n_sequence_val,n_sequence_test,
                  val_cutoff,test_cutoff):


    sample_list_train = list(range(0, int(len(X)*val_cutoff-n_observation_y-n_observation_X)))
    sample_list_val = list(range(int(len(X)*val_cutoff),int(len(X)*test_cutoff)))
    sample_list_test= list(range(int(len(X)*test_cutoff),int(len(X)-n_observation_y-n_observation_X)))

    random.shuffle(sample_list_train)
    random.shuffle(sample_list_val)
    random.shuffle(sample_list_test)

    X_train=np.zeros((n_sequence_train, n_observation_X, X.shape[1]))
    X_val=np.zeros((n_sequence_val, n_observation_X, X.shape[1]))
    X_test=np.zeros((n_sequence_test, n_observation_X, X.shape[1]))

    y_train=np.zeros((n_sequence_train, n_observation_y, 1))
    y_val=np.zeros((n_sequence_val, n_observation_y, 1))
    y_test=np.zeros((n_sequence_test, n_observation_y, 1))


    def create_sequence(X_,y_,sample_list,n_sequence):
        index=0
        for i in sample_list[0:n_sequence]:
            X_[index] = X.iloc[i:i + n_observation_X].values
            y_[index]= y.iloc[i + n_observation_X:i + n_observation_X + n_observation_y].values
            index=index+1
        return X_, y_

    X_train, y_train = create_sequence(X_train,y_train,sample_list_train,n_sequence_train)
    X_val, y_val = create_sequence(X_val,y_val,sample_list_val,n_sequence_val)
    X_test, y_test = create_sequence(X_test,y_test,sample_list_test,n_sequence_test)

    return X_train, X_val, X_test, y_train, y_val,y_test


n_observation_X=24 * 7*4  # For example, a week of data for the sequence
n_observation_y=24 # We would like to forecast the 24 prices of the next day during the auction of today
n_sequence_train=60
n_sequence_val=5
n_sequence_test=5
val_cutoff=0.8
test_cutoff=0.9


X_train, X_val, X_test, y_train, y_val,y_test = sequence_data(X, y,
                  n_observation_X, n_observation_y,
                  n_sequence_train,n_sequence_val,n_sequence_test,
                  val_cutoff,test_cutoff)

model = Sequential()
model.add(LSTM(units=64, input_shape=X_train.shape[1:], return_sequences=True))
model.add(LSTM(units=64, return_sequences=True))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dense(y.shape[1], activation='linear'))  # Output layer with 24 neurons (one for each hour)

# Compile and train the model
initial_learning=0.01 # Default value is 0.001
optimizer = Nadam(lr=initial_learning) #change the optimizer and right all the default value
model.compile(optimizer=optimizer, loss='mse',metrics=['mae'])
es = callbacks.EarlyStopping(patience=15, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1000, batch_size=16, callbacks=[es], validation_data=(X_val,y_val),verbose=1,shuffle=False)
model.save('model.h5')
