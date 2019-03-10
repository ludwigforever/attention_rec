# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import time
import random
import pandas as pd

import keras
from keras import backend as K
from keras import regularizers
from keras.regularizers import l2
from keras.models import Sequential, load_model, Model

from keras.layers import Dense, Activation, Dropout, Flatten,Average, Multiply, Dot, Input, Activation, Lambda, Flatten, Embedding,Concatenate, Layer,Reshape


from keras.layers import LSTM, recurrent
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.metrics import top_k_categorical_accuracy
from custom_layer import AttentionRNN, SelfAttention, Self_RNN\

class RecModel(object):
    def __init__(self, batch_size, max_length, n_hidden_units,
                 n_movies, n_genres, train_epochs, n_train_user, n_val_user):
        super(RecModel, self).__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_hidden_units = n_hidden_units
        self.n_movies = n_movies
        self.n_genres = n_genres
        self.train_epochs = train_epochs
        self.n_train_user = n_train_user
        self.n_val_user = n_val_user
        self.final_model=None

    def build(self):
        inputs = Input(shape=(self.max_length, self.n_movies + self.n_genres))
        embedding = Dense(self.n_hidden_units, activation='softmax', name='movie')(inputs)
        print(embedding.shape)
        # res = Reshape((max_length//slice_k,slice_k,n_hidden_units))(embedding)
        # print(embedding.shape)
        '''
        attention_out = AttentionRNN(self.n_hidden_units, input_shape=(self.max_length, self.n_hidden_units), dropout=0.2,
                                      name='attention')(embedding)
        '''
        self_rnn_out=Self_RNN(self.n_hidden_units, input_shape=(self.max_length,self.n_hidden_units), name='Self_RNN')(embedding)
        # print(attention_out.shape)
        # self_out = SelfAttention(n_hidden_units, name='self_attention')(embedding)
        # out=Concatenate(axis=-1)([attention_out,self_out])
        out = Dense(self.n_movies, activation='softmax', name='out')(self_rnn_out)
        self.final_model = Model(input=inputs, output=out)
        return self.final_model

    def modelCompile(self, model):
        model.summary()
        opti=keras.optimizers.Adam(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=[self.top_10_CCE])

    def train(self, model, gen_train, gen_val, scheduler):
        start_time = time.time()

        filepath = "model/model_{epoch:02d}-{val_top_10_CCE:.7f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_top_10_CCE', verbose=1, save_best_only=True, mode='max')
        reduce_lr_on_pl = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1,
                                                            mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
        # early_stopping = EarlyStopping(monitor='val_top_10_CCE', mode='max', patience=5, verbose=2)
        reduce_lr = LearningRateScheduler(scheduler)
        print('model training')
        # call_back_list=[early_stopping,checkpoint]
        call_back_list = [checkpoint, reduce_lr, reduce_lr_on_pl]
        model.fit_generator(generator=gen_train, epochs=self.train_epochs, steps_per_epoch=self.n_train_user,
                                 validation_data=gen_val, validation_steps=self.n_val_user, callbacks=call_back_list)
        # finalmodel.fit_generator(generator=gen_data,epochs=train_epochs,steps_per_epoch=n_train_user)
        end_time = time.time()
        print('total time:%f' % (end_time - start_time))
        
        
    def top_10_CCE(y_true,y_pred):
        return top_k_categorical_accuracy(y_true,y_pred,k=10)
