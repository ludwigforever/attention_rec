from prepare_data import DataHandler
import keras
from keras import backend as K
import time
import random
import pandas as pd

import keras
from keras import backend as K
from keras import regularizers
from keras.regularizers import l2
from keras.models import Sequential, load_model, Model

from keras.layers import Dense, Activation, Dropout, Flatten,Average, Multiply, Dot, Input, Activation, Lambda, Flatten, Embedding,Concatenate, Layer,Reshape


from keras.layers import LSTM, recurrent, Conv1D, GaussianNoise, GRU
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.metrics import top_k_categorical_accuracy
from custom_layer import multi_head, SelfAttention, LSTM_improve

def main():
    batch_size = 15
    n_movies = 3706
    n_users = 6040
    n_genres = 18
    n_usercode = 29
    max_length = 20
    n_hidden_units = 64
    train_epochs = 50

    dataset = DataHandler(batch_size, max_length, n_movies, n_genres, n_usercode)
    training_set, validation_set, n_train_user, n_val_user = dataset.get_train_data()
    #training_set, validation_set, n_train_user, n_val_user = dataset.get_train_data_lstm()

    def top_10_CCE(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=10)

    #inputs1 = Input(shape=(max_length, n_usercode))
    inputs = Input(shape=(max_length, n_movies + n_genres))# + n_usercode))
    noise = GaussianNoise(0.01)(inputs)
    embedding = Dense(n_hidden_units, activation='tanh', name='embedding')(noise)

    gru = GRU(n_hidden_units, name='GRU', return_sequences = True)(embedding)
    #lstm=LSTM(n_hidden_units, name='LSTM', return_sequences = True)(embedding)
    #lstm_improve = LSTM_improve(n_hidden_units, name='LSTM')(embedding)
    cov = Conv1D(n_hidden_units, 3, strides=1, padding='causal', data_format='channels_last', dilation_rate=1, activation='tanh', use_bias=True)(embedding)
    #multi = similar_RNN_multi(n_hidden_units, name='similar_RNN_multi')(embedding)
    # self_rnn_out = Self_RNN(n_hidden_units, name='Self_RNN')(embedding)
    #bi_self_rnn_out = Bi_Self_RNN(n_hidden_units, name='Bi_Self_RNN')(embedding)
    #similar = similar_RNN(n_hidden_units, name='similar_RNN')(embedding)
    #similar=genres_similar(n_hidden_units, name='genres_similar')(embedding)
    #weight=weight_RNN_multi(n_hidden_units, name='weight_RNN_multi')(embedding)
    lstm_cov=Concatenate(axis=-1, name='lstm_cov')([gru,cov])
    multi=multi_head(n_hidden_units, name='multi_head')(lstm_cov)
    #out = Activation('relu')(multi)

    out = Dense(n_movies, activation='softmax', name='out')(multi)

    finalmodel = Model(input=inputs, output=out)
    finalmodel.summary()
    '''
    opti=keras.optimizers.Adam(lr=0.005)
    finalmodel.compile(loss='categorical_crossentropy', optimizer=opti, metrics=[top_10_CCE])
    '''
    finalmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[top_10_CCE])
    #finalmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[top_10_CCE])

    start_time = time.time()

    filepath = "model_extend/model_{epoch:02d}-{val_top_10_CCE:.7f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_top_10_CCE', verbose=1, mode='max')
    reduce_lr_on_pl = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1,
                                                        mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    # early_stopping = EarlyStopping(monitor='val_top_10_CCE', mode='max', patience=5, verbose=2)
    print('model training')
    # call_back_list=[early_stopping,checkpoint]
    call_back_list = [checkpoint, reduce_lr_on_pl]
    finalmodel.fit_generator(generator=training_set, epochs=train_epochs, steps_per_epoch=n_train_user,
                        validation_data=validation_set, validation_steps=n_val_user, callbacks=call_back_list, verbose=1)
    # finalmodel.fit_generator(generator=gen_data,epochs=train_epochs,steps_per_epoch=n_train_user)
    end_time = time.time()
    print('total time:%f' % (end_time - start_time))


if __name__ == '__main__':
    main()
