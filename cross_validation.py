import argparse
import os 
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(description = 'prodlda cross validation wrapper')

parser.add_argument('--emb_path', type=str, default='', help='path to the embedding file')
parser.add_argument('--data_path', type=str, help='path to a fold of data')
parser.add_argument('--save_path', type=str, help='save path for every run')
parser.add_argument('--token_size', type=str, default='25709', help='number of words in vocabulary')
parser.add_argument('--maxlen', type=str, default='301', help='maximum tokens in a sentence')
parser.add_argument('--gpu', type=str, help='index of the gpu core which would contain this model')

args = parser.parse_args()

def run_script(params, fold):
    
    cmd = ('CUDA_VISIBLE_DEVICES={} python training.py'.format(args.gpu) +
          ' --fold ' + str(fold) +
          ' --emb_path ' + args.emb_path +
          ' --data_path ' + args.data_path +
          ' --save_path ' + args.save_path +
          ' --token_size ' + args.token_size + 
          ' --epochs ' + params['epochs'] +
          ' --lr ' + params['lr'] +
          ' --maxlen ' + args.maxlen +
          ' --mode train' + ' &')

    os.system(cmd)

#Hyperparameters
hyperparameters = {'epochs': ['100'],
                   'lr': ['3e-5']}

for params in ParameterGrid(hyperparameters):
    for fold in range(5): #Hard coded values of fold
        run_script(params, fold)
