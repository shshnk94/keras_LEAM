# -*- coding: utf-8 -*-
import pickle as pkl
import h5py
import argparse
import numpy as np
import os
from sklearn.model_selection import KFold
from keras.preprocessing.sequence import pad_sequences

parser = argparse.ArgumentParser(description='Preprocess pickle files into dataset')

parser.add_argument('--mode', type=str, help='cross validation or 1 fold validation')
parser.add_argument('--sampling', type=bool, help='flag to use sampling or the entire dataset')
parser.add_argument('--data_path', type=str, help='pickle file with data')
parser.add_argument('--emb_path', type=str, help='path to the file of glove embeddings')
parser.add_argument('--save_path', type=str, help='folder to store the dataset')


args = parser.parse_args()

with open(args.data_path, 'rb') as f:
    dataset = pkl.load(f)
    (train_text, 
     val_text, 
     test_text, 
     train_label, 
     val_label, 
     test_label, 
     dictionary, 
     reverse_dictionary) = dataset

# Train, validation, and test set size
print("Training set: ", len(train_text))
print("Validation set: ", len(val_text))
print("Test set: ", len(test_text))

# max_len
max_len = max([len(i) for i in train_text])
max_len = max([len(i) for i in test_text]) if max([len(i) for i in test_text]) > max_len else max_len
max_len = max([len(i) for i in val_text]) if max([len(i) for i in val_text]) > max_len else max_len
print("Maximum sentence length: ", max_len)

#Sample if necessary.
if args.sampling:

    #Generates a sample of 10% the dataset size.
    indices = np.random.choice(len(train_text), int(0.1 * len(train_text)))
    train_text = [train_text[index] for index in indices]
    train_label = [train_label[index] for index in indices]

#Add padding to vocabulary
reverse_dictionary['PAD'] = len(dictionary)
dictionary[len(dictionary)] = 'PAD'
with open(args.emb_path, 'rb') as f:
    embeddings = pkl.load(f)
    embeddings = np.vstack((embeddings, np.zeros((1, embeddings.shape[1]))))

with open(os.path.join(args.data_path.split('/')[0], 'embeddings.pkl'), 'wb') as f:
    pkl.dump(embeddings, f)    

def store(x, y, path, mode):
    
    #Since the vocabulary is indexed from 0, padding index is moved to length_of_vocabulary. 
    x = pad_sequences(x, maxlen=max_len, dtype="int32", padding='post', value=reverse_dictionary['PAD']) 
    y = np.array(y).squeeze()


    with h5py.File(os.path.join(path, mode + '.h5'), 'w') as handle:
        handle.create_dataset('x', data=x)
        handle.create_dataset('y', data=y)

if args.mode != 'cv':
    
    store(train_text, train_label, args.save_path, 'train')
    store(val_text, val_label, args.save_path, 'valid') 
    
else:

    kf = KFold(n_splits=5)

    for fold, indices in enumerate(kf.split(train_text)):

        fold_path = os.path.join(args.save_path, 'fold{}'.format(fold))
        if not os.path.isdir(fold_path):
            os.makedirs(fold_path)
        
        train_text_fold, train_label_fold = [train_text[i] for i in indices[0]], [train_label[i] for i in indices[0]] 
        val_text_fold, val_label_fold = [train_text[i] for i in indices[1]], [train_label[i] for i in indices[1]] 
        
        store(train_text, train_label, fold_path, 'train')
        store(val_text, val_label, fold_path, 'valid') 

store(test_text, test_label, args.save_path, 'test')
