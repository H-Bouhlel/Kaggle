import re
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical




#I choose to build 3 hidden layers

def model(len_num,features):
    model = Sequential()
    model.add(Embedding(len_num, 100, input_length=features))
    model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2 ))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model