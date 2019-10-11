# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.preprocessing import sequence

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils.vis_utils import plot_model
import pickle
from keras import backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import *

#en
BATCHSIZE = 16
sentence_size = 15
maxlen = 146 # Length of the sentence with maximum length in the dataset
embedding_size = 300
token_size = 856701 # No. words in vocabulary obtained from the corpus
class_num = 10 # No. class labels as per the original LEAM codebase
VECTOR_DIR = 'dataset/en_vector_google.pkl'

from model import *
from utils import get_minibatches_idx, restore_from_save, tensors_key_in_file, prepare_data_for_emb, load_class_embedding
from keras.engine.topology import Layer

class Options(object):
    def __init__(self):
        self.fix_emb = True
        self.restore = False
        self.W_emb = None
        self.W_class_emb = None
        self.maxlen = maxlen
        self.n_words = None
        self.embed_size = embedding_size
        self.lr = 1e-3
        self.batch_size = BATCHSIZE
        self.dropout = 0.5
        self.part_data = False
        self.portion = 1.0
        self.clip_grad = None
        self.class_penalty = 1.0
        self.ngram = 55
        self.H_dis = 300

    #Iterating along the class attributes
    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value

opt = Options()

def emb_classifier(x_emb, x_mask, y,W_class, dropout=0.5, opt=opt):
    W_class_tran = tf.transpose(W_class, [0,2,1]) # b* e * c
    x_emb = tf.expand_dims(x_emb, 3)  # b * s * e * 1
    H_enc = att_emb_ngram_encoder_cnn(x_emb, x_mask, W_class, W_class_tran, opt)
    print(H_enc.get_shape().as_list())
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
        token_seq = x[0]
        y_seq = x[1]
        token_mask = x[2]
        class_all = x[3]
        mask_seq = tf.ones([K.shape(token_seq)[0],K.shape(token_seq)[1]])
        rep = emb_classifier(token_seq, mask_seq,y_seq,class_all)
        return rep

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], class_num)

def getdata():
    import pickle
    train_set = open('dataset/enDataTrain.pkl','rb')
    train_x = pickle.load(train_set)
    print("Training set: ", len(train_x))
    train_y = np.array(pickle.load(train_set))
	
    test_set = open('dataset/enDataTest.pkl','rb')
    test_x = pickle.load(test_set)
    print("Test set: ", len(test_x))
    test_y = np.array(pickle.load(test_set))

    class_all = np.array(range(0,class_num))

    return train_x,train_y,test_x,test_y,class_all

def train():

    # Read word embeddings from VECTOR_DIR
    f = open(VECTOR_DIR, 'rb')
    word_vector = np.array(pickle.load(f))
    f.close()
	
    # f0 - Where you convert the text sequence into their respective embeddings.
    sentence_inputs = Input(shape=(maxlen,), dtype='int32')
    print("sentence_inputs, each of size max_len: ", K.int_shape(sentence_inputs))
    sentence_embeddings = Embedding(token_size, embedding_size,mask_zero=False,weights=[word_vector],trainable=False)(sentence_inputs)
    print("sentence_embeddings, each of shape (max_len, embedding_size): ", K.int_shape(sentence_embeddings))

    # Produces attention values for each token in a sequence (Not mentioned in the original paper)
    sentence_attn = AttentionLayer()(sentence_embeddings)
    sentence_encoder = Model(sentence_inputs,sentence_attn)

    token_inputs = Input(shape=(sentence_size,maxlen,), dtype='int32')
    label_inputs = Input((class_num,), dtype='int32')

    # Obtain the class embedding C (K X P) = (20 X 300)
    class_all_inputs = Input((class_num,), dtype='int32')
    class_all_embeddings = Embedding(class_num, embedding_size,mask_zero=False)(class_all_inputs)

    token_encoder = TimeDistributed(sentence_encoder)(token_inputs)
	
    # f1 layer which outputs 'z' (average of the word embeddings weighted by the attentions score).
    doc_leam = LEAM()([token_encoder,label_inputs,token_inputs,class_all_embeddings])

    # f2 layer (output) where you get the class probability after taking the sentence embedding.
    output = Dense(class_num,activation='softmax')(doc_leam)

    model = Model(input=[token_inputs,label_inputs,class_all_inputs], output=[output])
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    print(model.summary())

    train_x,train_y,test_x,test_y,class_all = getdata()
    print('-------------------------')
    model.fit([train_x,train_y,np.repeat(class_all[np.newaxis,:],len(train_x),axis=0)], train_y,epochs=10,batch_size=BATCHSIZE,validation_data=([test_x,test_y,np.repeat(class_all[np.newaxis,:],len(test_x),axis=0)], test_y))
    model.save('baseline.h5')


if __name__ == '__main__':
    train()


