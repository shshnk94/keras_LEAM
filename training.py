# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import pickle
import h5py
import os

import tensorflow as tf
import keras
from keras.preprocessing import sequence
from keras.backend.tensorflow_backend import set_session
from keras.utils.np_utils import to_categorical
#from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.models import Model
from keras.layers import *
from keras.engine.topology import Layer

from model import *
from utils import restore_from_save, tensors_key_in_file, prepare_data_for_emb, load_class_embedding, datagen

parser = argparse.ArgumentParser(description='LEAM implementation in Keras')

parser.add_argument('--fold', type=str, default='', help='current cross valid fold number')
parser.add_argument('--data_path', type=str, default='dataset/', help='directory containing data')
parser.add_argument('--emb_path', type=str, default='dataset/en_vector_google.pkl', help='directory containing word embeddings')
parser.add_argument('--save_path', type=str, default='results/', help='path to save results')

parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--embedding_size', type=int, default=300, help='dimension of embeddings')
parser.add_argument('--sentence_size', type=int, default=15, help='dimension of embeddings')
parser.add_argument('--maxlen', type=int, default=146, help='maximum length of a sentence')
parser.add_argument('--token_size', type=int, default=856701, help='vocabulary size')
parser.add_argument('--class_num', type=int, default=2, help='number of classification classes')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--mode', type=str, default='train', help='train or eval model')
#parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

args = parser.parse_args()

class Options(object):
    def __init__(self):
        self.fix_emb = True
        self.restore = False
        self.W_emb = None
        self.W_class_emb = None
        self.maxlen = args.maxlen
        self.n_words = None
        self.embed_size = args.embedding_size
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.dropout = 0.5
        self.part_data = False
        self.portion = 1.0
        self.clip_grad = None
        self.class_penalty = 1.0
        self.ngram = 55
        self.H_dis = 300
        self.class_num = args.class_num

    #Iterating along the class attributes
    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value

opt = Options()

def emb_classifier(x_emb, x_mask, W_class, dropout=0.5, opt=opt):

    W_class_tran = tf.transpose(W_class, [0,2,1]) # b* e * c
    x_emb = tf.expand_dims(x_emb, 3)  # b * s * e * 1
    H_enc = att_emb_ngram_encoder_cnn(x_emb, x_mask, W_class, W_class_tran, opt)

    H_enc_list= tf.unstack(H_enc, axis=-1)
    # print(H_enc_list.shape)
    logits_list = []
    for i, ih in enumerate(H_enc_list):
        logits_list.append(discriminator_0layer(ih, opt, dropout, prefix='classify_{}'.format(i), num_outputs=1, is_reuse=False) )

    logits = tf.concat(logits_list,-1)
    return logits

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        print("Attention layer weights and bias shapes: ", K.shape(self.W), K.shape(self.b))
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        x = K.permute_dimensions(inputs, (0, 2, 1))
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

class LEAM(Layer):

    def __init__(self, **kwargs):

        self.supports_masking = False
        self.is_train = tf.constant(True, dtype=tf.bool)
        super(LEAM, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def call(self, x, mask=None):

        #token_seq, y_seq, token_mask, class_all = x
        token_seq, class_all = x
        mask_seq = tf.ones([K.shape(token_seq)[0],K.shape(token_seq)[1]])
        rep = emb_classifier(token_seq, mask_seq, class_all)

        return rep

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], args.class_num)

def getgen(mode):
    
    if args.mode == 'train':
        
        args.data_path = os.path.join(args.data_path, 'fold{}'.format(args.fold)) if args.fold != '' else args.data_path
        train_handle = h5py.File(os.path.join(args.data_path, 'train.h5'), 'r')
        valid_handle = h5py.File(os.path.join(args.data_path, 'valid.h5'), 'r')
        
        return train_handle, valid_handle
 
    else:

        test_handle = h5py.File(os.path.join(args.data_path, 'test.h5'))
   
        return test_handle

def train():

    # Read word embeddings from VECTOR_DIR
    with open(args.emb_path, 'rb') as f:
        word_vector = np.array(pickle.load(f))
	
    # f0 - Where you convert the text sequence into their respective embeddings.
    sentence_inputs = Input(shape=(args.maxlen,), dtype='int32')
    print("sentence_inputs, each of size max_len: ", K.int_shape(sentence_inputs))
    sentence_embeddings = Embedding(args.token_size + 1, args.embedding_size, mask_zero=False, weights=[word_vector], trainable=False)(sentence_inputs)
    print("sentence_embeddings, each of shape (max_len, embedding_size): ", K.int_shape(sentence_embeddings))

    # Calculates the attention values \beta and then the sentence encoder - z.
    #sentence_attn = AttentionLayer()(sentence_embeddings)
    #sentence_encoder = Model(sentence_inputs,sentence_attn)

    # Obtain the class embedding C (K X P) = (20 X 300)
    class_all_inputs = Input((args.class_num,), dtype='int32')
    class_all_embeddings = Embedding(args.class_num, args.embedding_size,mask_zero=False)(class_all_inputs)

    #token_inputs = Input((args.sentence_size, args.maxlen,), dtype='int32')
    #label_inputs = Input((args.class_num,), dtype='int32')

    #token_encoder = TimeDistributed(sentence_encoder)(token_inputs)
	
    # f1 layer which outputs 'z' (average of the word embeddings weighted by the attentions score).
    #doc_leam = LEAM()([token_encoder, label_inputs, token_inputs, class_all_embeddings])
    doc_leam = LEAM()([sentence_embeddings, class_all_embeddings])

    # f2 layer (output) where you get the class probability after taking the sentence embedding - z (doc_leam here)
    output = Dense(args.class_num, activation='softmax')(doc_leam)

    #model = Model(input=[token_inputs,label_inputs,class_all_inputs], output=[output])
    model = Model(input=[sentence_inputs, class_all_inputs], output=[output])
    #plot_model(model, to_file=os.path.join(args.save_path, 'model_plot.png'), show_shapes=True, show_layer_names=True)

    optimizer = keras.optimizers.Adam(lr=args.lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    print(model.summary())

    #train_x, train_y, test_x, test_y, class_all = getdata(args.mode)
    train_handle, valid_handle = getgen(args.mode)
    
    history = model.fit_generator(datagen(train_handle, opt),
                                  epochs=args.epochs,
                                  steps_per_epoch=np.ceil(train_handle['x'].shape[0] / args.batch_size),
                                  validation_data = datagen(valid_handle, opt),
                                  validation_steps=np.ceil(valid_handle['x'].shape[0] / args.batch_size))
    
    #Save the cross_validation results
    args.save_path = os.path.join(args.save_path, 'e_{}_lr{}'.format(args.epochs, args.lr))
    args.save_path = os.path.join(args.save_path, 'fold{}'.format(args.fold)) if args.fold != '' else args.save_path

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    with open(os.path.join(args.save_path, 'accuracy.pkl'), 'wb') as f:
        pickle.dump(history.history['acc'], f)

    model.save(os.path.join(args.save_path, 'baseline.h5'))

def test():
    
    #test_x, test_y, class_all = getdata(args.mode)
    test_handle = getgen(args.mode)
    model = keras.models.load_model(os.path.join(args.save_path, 'baseline.h5'), custom_objects={'LEAM': LEAM})
    result = model.evaluate([test_x, test_y, np.repeat(class_all[np.newaxis,:],len(test_x),axis=0)],
                            test_y,
                            batch_size=args.batch_size) 

    print("Result on the held-out set: ", result)

if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True 
    config.log_device_placement = True
    sess = tf.Session(config=config)
    set_session(sess)

    train() if args.mode == 'train' else test()
