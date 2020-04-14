"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

import os
import numpy as np 
from glob import glob
import random
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import cv2
from tqdm.notebook import tqdm
from rdkit import Chem
from rdkit import RDLogger

log = RDLogger.logger()
log.setLevel(RDLogger.CRITICAL)

#--------------------------------------------------------------------------------------------------------------------------------------------------
def LOG_INFO(log_text,p_color='green',rep=True):
    if rep:
        print(colored('#    LOG:','blue')+colored(log_text,p_color))
    else:
        print(colored('#    LOG:','blue')+colored(log_text,p_color),end='\r')

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
#--------------------------------------------------------------------------------------------------------------------------------------------------
# Tokenizer Class
'''
* The tokenizer class is kept as same as the  State Of The Art **LSTM_Chem** : **smiles_tokenizer** 
( https://github.com/topazape/LSTM_Chem/blob/b469e6aff2df28de99b9c4541a1a76d4d4fd77b0/lstm_chem/utils/smiles_tokenizer2.py)**
'''
class SmilesTokenizer(object):
    def __init__(self):
        atoms = ['Al', 'As', 'B', 'Br', 'C', 'Cl', 
                 'F', 'H', 'I', 'K', 'Li', 'N',
                 'Na', 'O', 'P', 'S', 'Se', 'Si', 'Te']
        special = ['(', ')', '[', ']', '=', '#', '%', '0', '1', '2', '3', '4', '5',
            '6', '7', '8', '9', '+', '-', 'se', 'te', 'c', 'n', 'o', 's']
        padding = ['G', 'A', 'E']

        self.table = sorted(atoms, key=len, reverse=True) + special + padding
        table_len = len(self.table)

        self.table_2_chars = list(filter(lambda x: len(x) == 2, self.table))
        self.table_1_chars = list(filter(lambda x: len(x) == 1, self.table))

        self.one_hot_dict = {}
        for i, symbol in enumerate(self.table):
            vec = np.zeros(table_len, dtype=np.float32)
            vec[i] = 1
            self.one_hot_dict[symbol] = vec

    def tokenize(self, smiles):
        smiles = smiles + ' '
        N = len(smiles)
        token = []
        i = 0
        while (i < N):
            c1 = smiles[i]
            c2 = smiles[i:i + 2]

            if c2 in self.table_2_chars:
                token.append(c2)
                i += 2
                continue

            if c1 in self.table_1_chars:
                token.append(c1)
                i += 1
                continue

            i += 1

        return token

    def one_hot_encode(self, tokenized_smiles):
        result = np.array(
            [self.one_hot_dict[symbol] for symbol in tokenized_smiles],
            dtype=np.float32)
        result = result.reshape(1, result.shape[0], result.shape[1])
        return result

#--------------------------------------------------------------------------------------------------------------------------------------------------
# Tfrecord utils
'''
* one hot encodes the **tokenized-padded** smiles data
* encodes the data to **.png** format
* extacts the **bytes** and writes to **tfrecord**
'''
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def to_tfrecord(data,save_dir,r_num,tokenizer):
    tfrecord_name='{}.tfrecord'.format(r_num)
    tfrecord_path=os.path.join(save_dir,tfrecord_name)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        for tp_smi in data:
            # one_hot_encoding
            X = np.array([tokenizer.one_hot_dict[symbol] for symbol in tp_smi[:-1]],dtype=np.uint8)
            y = np.array([tokenizer.one_hot_dict[symbol] for symbol in tp_smi[1:]],dtype=np.uint8)
            # png encoded data
            _,X_coded = cv2.imencode('.png',X)
            _,y_coded = cv2.imencode('.png',y)
            # byte conversion
            x_bytes = X_coded.tobytes()
            y_bytes = y_coded.tobytes()
            # data_dict
            data_dict ={  'X':_bytes_feature(x_bytes),
                          'Y':_bytes_feature(y_bytes)}
            # writting to tfrecord
            features=tf.train.Features(feature=data_dict)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)   

# save the data as tfrecord
def create_records(_smiles,save_path,mode,tokenizer,BATCH_SIZE=10240):
    print('Saving The data to tfrecord:',mode)
    for i in  tqdm(range(0,len(_smiles),BATCH_SIZE)):
        data=_smiles[i:BATCH_SIZE+i]
        r_num=i//BATCH_SIZE
        to_tfrecord(data,save_path,r_num,tokenizer)

# dataset function
def data_input_fn(data_path,mode,DATA_DIM,BATCH_SIZE,BUFFER_SIZE): 
    def _parser(example):
        feature ={  'X'  : tf.io.FixedLenFeature([],tf.string) ,
                    'Y'  : tf.io.FixedLenFeature([],tf.string)}    
        parsed_example=tf.io.parse_single_example(example,feature)
        inp=parsed_example['X']
        inp=tf.image.decode_png(inp,channels=1)
        inp=tf.cast(inp,tf.float32)
        inp=tf.reshape(inp,DATA_DIM)
        
        tgt=parsed_example['Y']
        tgt=tf.image.decode_png(tgt,channels=1)
        tgt=tf.cast(tgt,tf.float32)
        tgt=tf.reshape(tgt,DATA_DIM)
        
        return inp,tgt
    # get tfrecords from data_path
    _pattern=os.path.join(data_path,mode,'*.tfrecord')
    # extract file paths
    file_paths = tf.io.gfile.glob(_pattern)
    # create data set with standard parsing,shuffling,batching and prefetching
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(_parser)
    dataset = dataset.shuffle(BUFFER_SIZE,reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
#--------------------------------------------------------------------------------------------------------------------------------------------------
def check_smi(smi):
    '''
    Check If an smi is chemically valid
    Returns None if not valid
    '''
    m = Chem.MolFromSmiles(smi,sanitize=False)
    if m is None:
        print('invalid SMILE:',smi)
        return None
    else:
        try:
            Chem.SanitizeMol(m)
            return smi
        except:
            print('invalid chemistry:',smi)
            return None
#--------------------------------------------------------------------------------------------------------------------------------------------------
