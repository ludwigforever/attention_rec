# -*- coding: utf-8 -*-
import numpy as np
import random
import csv    #加载csv包便于读取csv文件

import sys


class DataHandler(object):
    def __init__(self, batch_size, max_length, n_movies, n_genres):
        super(DataHandler, self).__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_movies = n_movies
        self.n_genres = n_genres
        self.tag_map = {'Sci-Fi': 0, 'Documentary': 1, 'War': 2, 'Romance': 3, 'Mystery': 4, 'Western': 5, 'Comedy': 6, 'Musical': 7, 'Thriller': 8, 'Action': 9, "Children's": 10, 'Horror': 11, 'Crime': 12, 'Fantasy': 13, 'Drama': 14, 'Animation': 15, 'Adventure': 16, 'Film-Noir': 17}




    def load_data(self, path):
        data = []
        with open(path, 'r') as f:
            for sequence in f:
                data.append(sequence)

        n_user = len(data)
        return data, n_user

    def sequence_generator(self, lines):

        while True:
            for j, sequence in enumerate(lines):
                # sequence=[]
                # print(sequence)
                sequence = sequence.split()
                user_id = sequence[0]
                sequence = sequence[1:]
                # !!!!!!!修改
                sequence = [[int(sequence[2 * i]), str(sequence[2 * i + 1])] for i in range(int(len(sequence) / 2))]
                # sequence = [[int(sequence[2*i]), str(sequence[2*i + 1]), int(sequence[2*i + 2])] for i in range(int(len(sequence) / 3))]
                yield sequence, user_id

    def tag2code(self, tag):
        tag_encoding = np.zeros(self.n_genres)
        keys = tag.split('|')
        ids = []
        for k in keys:
            ids.append(self.tag_map[k])
        tag_encoding[ids] = 1
        return tag_encoding

    def get_features(self, movie_in):
        movie_id, genres = movie_in[0], movie_in[1]
        one_hot_encoding = np.zeros(self.n_movies)
        one_hot_encoding[movie_id] = 1
        return np.concatenate((one_hot_encoding, self.tag2code(genres)))

    def prepare_input(self, sequences):

        X = np.zeros((self.batch_size, self.max_length, self.n_movies + self.n_genres), dtype='float')
        Y = np.zeros((self.batch_size, self.n_movies), dtype='float')
        for i, sequence in enumerate(sequences):
            user_id, in_seq, target = sequence
            '''print('prepare_input:user_id')
            print(user_id)
            print('prepare_input:in_seq')
            print(in_seq)
            print('prepare_input:target')
            print(target)'''
            seq_features = np.array(list(map(lambda x: self.get_features(x), in_seq)))
            X[i, self.max_length - len(in_seq):, :] = seq_features
            one_hot_encoding = np.zeros(self.n_movies)
            one_hot_encoding[target[0]] = 1
            Y[i] = one_hot_encoding
        return (X, Y)

    def gen_mini_batch(self, sequence_generator, test=False, max_reuse_sequence=np.inf):
        batch_size = 16
        while True:
            j = 0
            sequences = []

            if test:
                batch_size = 1
            while j < batch_size:

                sequence, user_id = next(sequence_generator)
                # finds the lengths of the different subsequences
                if not test:
                    # print('next')
                    # print(sequence)
                    seq_lengths = sorted(random.sample(range(2, len(sequence) - 1),
                                                       min([batch_size - j,
                                                            len(sequence) - 2,
                                                            max_reuse_sequence])))
                else:
                    seq_lengths = [int(len(sequence) / 2)]

                skipped_seq = 0
                for l in seq_lengths:
                    start = max(0, l - self.max_length)  # sequences cannot be longer than self.max_lenght
                    target = sequence[l]
                    sequences.append([user_id, sequence[start:l], target])
                j += len(seq_lengths)

            if test:
                yield self.prepare_input(sequences), [i[0] for i in sequence[seq_lengths[0]:]]
            else:
                yield self.prepare_input(sequences)

    def get_train_data(self):
        train_data, n_train_user = self.load_data('./data/train_set_sequences')
        val_data, n_val_user = self.load_data('./data/val_set_sequences')
        training_set = self.gen_mini_batch(self.sequence_generator(train_data))
        validation_set = self.gen_mini_batch(self.sequence_generator(val_data))
        return training_set, validation_set, n_train_user, n_val_user


