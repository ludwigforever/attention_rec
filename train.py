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


from keras.layers import LSTM, recurrent
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.metrics import top_k_categorical_accuracy
from custom_layer import AttentionRNN, SelfAttention, Bi_Self_RNN

def main():
    batch_size = 15
    n_movies = 3706
    n_users = 6040
    n_genres = 18
    n_usercode = 29
    max_length = 20
    n_hidden_units = 80
    train_epochs = 40

    dataset = DataHandler(batch_size, max_length, n_movies, n_genres, n_usercode)
    training_set, validation_set, n_train_user, n_val_user = dataset.get_train_data()

    def top_10_CCE(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=10)

    inputs = Input(shape=(max_length+1, n_movies + n_genres + n_usercode))
    embedding = Dense(n_hidden_units, activation='tanh', name='embedding')(inputs)

    lstm=LSTM(n_hidden_units, name='LSTM', return_sequences = False)(embedding)
    # self_rnn_out = Self_RNN(n_hidden_units, name='Self_RNN')(embedding)
    #bi_self_rnn_out = Bi_Self_RNN(n_hidden_units, name='Self_RNN')(embedding)

    out = Dense(n_movies, activation='softmax', name='out')(lstm)

    finalmodel = Model(input=inputs, output=out)
    finalmodel.summary()
    finalmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[top_10_CCE])

    start_time = time.time()

    filepath = "model_extend/model_{epoch:02d}-{val_top_10_CCE:.7f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_top_10_CCE', verbose=1, save_best_only=True, mode='max')
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
