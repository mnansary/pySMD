"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function

import os
import numpy as np 
import time
import tensorflow as tf
#--------------------------------------------------------------------------------------------------------------------------------------------------
# TODO: Add Stacked Structre

#--------------------------------------------------------------------------------------------------------------------------------------------------

def LSTM_Chem(units,n_table):
    '''
    The model is based on:Generative Recurrent Networks for De Novo Drug Design (https://onlinelibrary.wiley.com/doi/full/10.1002/minf.201700111)
    Adapted for TPU from: https://github.com/topazape/LSTM_Chem
    '''
    weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.05)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=units,
                                input_shape=(None, n_table),
                                return_sequences=True,
                                kernel_initializer=weight_initializer,
                                dropout=0.3))
    model.add(tf.keras.layers.LSTM(units=units,
                                input_shape=(None, n_table),
                                return_sequences=True,
                                kernel_initializer=weight_initializer,
                                dropout=0.5))
    model.add(tf.keras.layers.Dense(units=n_table,activation='softmax',kernel_initializer=weight_initializer))
    return model
#--------------------------------------------------------------------------------------------------------------------------------------------------
# model utils
'''
based on: https://github.com/topazape/LSTM_Chem
'''
def generate(model,sequence,tokenizer,MAX_LEN=128):
    while (sequence[-1] != 'E') and (len(tokenizer.tokenize(sequence)) <= MAX_LEN):
        x = tokenizer.one_hot_encode(tokenizer.tokenize(sequence))
        preds = model.predict_on_batch(x)[0][-1]
        next_idx = sample_with_temp(preds)
        sequence += tokenizer.table[next_idx]
        
    sequence = sequence[1:].rstrip('E')
    return sequence

def sample_with_temp(preds,sampling_temp=0.75):
  streched = np.log(preds) / sampling_temp
  streched_probs = np.exp(streched) / np.sum(np.exp(streched))
  return np.random.choice(range(len(streched)), p=streched_probs)