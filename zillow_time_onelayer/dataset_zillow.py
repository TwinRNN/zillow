# encoding:utf-8
import numpy as np
import pickle
import argparse
import os
import tensorflow as tf


class Dataset(object):

    def __init__(self, batch_size, seq_len, time_series_params):
        # time_series starting on 2014.12.28
        # transation date starting on 2016.1.1
        MAP = 369
        # print('read features')
        parser = argparse.ArgumentParser()

        # 配置训练数据的地址
        parser.add_argument('--buckets', type=str,
                            default='Data', help='input data path')
        # 配置模型保存地址
        # parser.add_argument('--output_dir', type=str,
        # default='', help='output model path')

        FLAGS, _ = parser.parse_known_args()
        train_file_path = os.path.join(FLAGS.buckets, "zillow-model-data-original")
        train_file_path2 = os.path.join(FLAGS.buckets, "model_time_day")
        # output_file = os.path.join(FLAGS.output_dir,"output.txt")
        # print("XXXXXXX:", FLAGS.data_dir, train_file_path)

        with tf.gfile.Open(train_file_path, 'rb') as f:
            raw_data = f.read()
            data = pickle.loads(raw_data)
            self.train_df = data['train_df'][: 150000]
            self.logerror_df = data['logerror_df'][: 150000]
            self.transactiondate_df = data['transactiondate_df'][: 150000]
            # print('train_df shape:', self.train_df.shape)
            # print('logerror_df shape:', self.logerror_df.shape)
            # print('transactiondate_df shape:', self.transactiondate_df.shape)

        with tf.gfile.Open(train_file_path2, 'rb') as f:
            raw_data = f.read()
            data = pickle.loads(raw_data)
            self.macro = data['macro']
            self.google = self.macro[:, : 15]
            self.economy = self.macro[:, 15: 23]
            self.tweet = self.macro[:, 23: 29]
            self.tweet_re = self.macro[:, 29: 35]

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.time_series_params = time_series_params

        self.time_series_step = time_series_params['time_series_step']
        delay_google = time_series_params['delay_google']
        delay_tweeter = time_series_params['delay_tweeter']
        delay_macro = time_series_params['delay_macro']
        delay_tweeter_re = time_series_params['delay_tweeter_re']

        self.move_google = MAP - self.time_series_step - delay_google
        self.move_tweeter = MAP - self.time_series_step - delay_tweeter
        self.move_macro = MAP - self.time_series_step - delay_macro
        self.move_tweeter_re = MAP - self.time_series_step - delay_tweeter_re
        self.epochs = 0
        self.cursor = 0
        self.size = 0

    def split(self, forward_step, train_num, valid_num, test_num, model_num=0):

        train_end = train_num + model_num * forward_step
        self.train = self.train_df[: train_end].values
        self.tr_logerror = self.logerror_df[: train_end].values
        self.tr_transactiondate = self.transactiondate_df[: train_end].values

        valid_end = train_end + valid_num
        self.valid = self.train_df[train_end: valid_end].values
        self.valid_logerror = self.logerror_df[train_end: valid_end].values
        self.valid_transactiondate = self.transactiondate_df[
            train_end: valid_end].values

        test_end = valid_end + test_num
        self.test = self.train_df[valid_end: test_end].values
        self.test_logerror = self.logerror_df[valid_end: test_end].values
        self.test_transactiondate = self.transactiondate_df[
            valid_end: test_end].values

        self.size = self.train.shape[0]

    def feature_size(self):
        feature_size = {'event': self.train_df.shape[
            1], "time": self.macro.shape[1]}
        return feature_size

    def next_batch(self,):
        """
        used in training phase
        :param batch_size:
        :param sequence_length:
        :param delay: for event taking place on D, we use time series date [D - time_series - delay: D - delay]
        :param time_series: length of historical time series data
        :return: next batch's data
        """
        data_set = self.batch_size * self.seq_len

        if self.cursor + data_set > self.size:
            self.epochs += 1
            self.cursor = 0
            self.roll(self.batch_size, self.seq_len)

        train = self.train[self.cursor: self.cursor + data_set]
        tr_logerror = self.tr_logerror[self.cursor: self.cursor + data_set]
        macro_features_google = []
        macro_features_macro = []
        macro_features_tweet = []
        macro_features_tweet_re = []

        for i in range(data_set):
            init_time_google = self.tr_transactiondate[i + self.cursor] + self.move_google
            init_time_macro = self.tr_transactiondate[i + self.cursor] + self.move_macro
            init_time_tweet = self.tr_transactiondate[i + self.cursor] + self.move_tweeter
            init_time_tweet_re = self.tr_transactiondate[i + self.cursor] + self.move_tweeter_re

            for _ in range(self.time_series_step):
                macro_features_google.append(self.google[init_time_google + _])
                macro_features_macro.append(self.economy[init_time_macro + _])
                macro_features_tweet.append(self.tweet[init_time_tweet + _])
                macro_features_tweet_re.append(self.tweet[init_time_tweet_re + _])
        # print('google_shape: ', np.array(macro_features_google).shape)
        # print('tweeter_shape: ', np.array(macro_features_tweet).shape)
        # print('macro_shape: ', np.array(macro_features_macro).shape)
        macro_features = np.concatenate((macro_features_tweet, macro_features_tweet_re,
                                         macro_features_macro, macro_features_google), axis=1)
        macro_features = np.asarray(macro_features).reshape(
            [self.batch_size, self.seq_len, self.time_series_step, -1])

        self.cursor += data_set

        train = train.reshape([self.batch_size, self.seq_len, -1])
        tr_logerror = tr_logerror.reshape([self.batch_size, self.seq_len, -1])

        # seqlen = [sequence_length for _ in range(batch_size)]
        # return train, tr_logerror, seqlen, macro_features, self.epochs
        # print macro_features.shape
        return train, tr_logerror, macro_features

    def valid_data(self):
        valid = self.valid[np.newaxis]
        seqlen = valid.shape[1]
        valid_logerror = self.valid_logerror.reshape((1, -1, 1))

        macro_features_google = []
        macro_features_macro = []
        macro_features_tweet = []
        macro_features_tweet_re = []

        for i in range(seqlen):
            date_in_event = self.valid_transactiondate[i]
            init_time_google = date_in_event + self.move_google
            init_time_macro = date_in_event + self.move_macro
            init_time_tweet = date_in_event + self.move_tweeter
            init_time_tweet_re = date_in_event + self.move_tweeter_re

            for _ in range(self.time_series_step):
                macro_features_google.append(self.google[init_time_google + _])
                macro_features_macro.append(self.economy[init_time_macro + _])
                macro_features_tweet.append(self.tweet[init_time_tweet + _])
                macro_features_tweet_re.append(self.tweet[init_time_tweet_re + _])
        macro_features = np.concatenate((macro_features_tweet, macro_features_tweet_re,
                                         macro_features_macro, macro_features_google), axis=1)
        macro_features = np.asarray(macro_features).reshape([1, seqlen, self.time_series_step, -1])
        return valid, valid_logerror, macro_features

    def test_data(self):
        test = self.test[np.newaxis]
        seqlen = test.shape[1]
        test_logerror = self.test_logerror.reshape((1, -1, 1))
        macro_features_google = []
        macro_features_macro = []
        macro_features_tweet = []
        macro_features_tweet_re = []

        for i in range(seqlen):
            date_in_event = self.test_transactiondate[i]
            init_time_google = date_in_event + self.move_google
            init_time_macro = date_in_event + self.move_macro
            init_time_tweet = date_in_event + self.move_tweeter
            init_time_tweet_re = date_in_event + self.move_tweeter_re

            for _ in range(self.time_series_step):
                macro_features_google.append(self.google[init_time_google + _])
                macro_features_macro.append(self.economy[init_time_macro + _])
                macro_features_tweet.append(self.tweet[init_time_tweet + _])
                macro_features_tweet_re.append(self.tweet[init_time_tweet_re + _])

        macro_features = np.concatenate((macro_features_tweet, macro_features_tweet_re,
                                         macro_features_macro, macro_features_google), axis=1)
        macro_features = np.asarray(macro_features).reshape([1, seqlen, self.time_series_step, -1])
        return test, test_logerror, macro_features

    def roll(self, batch_size, sequence_length):
        # 对数据进行旋转，保持rnn状态的联系性
        self.train = np.roll(self.train, batch_size * sequence_length, axis=0)
        self.tr_logerror = np.roll(
            self.tr_logerror, batch_size * sequence_length, axis=0)
        self.tr_transactiondate = np.roll(
            self.tr_transactiondate, batch_size * sequence_length, axis=0)
        # np.random.shuffle(self.X_train)
        # np.random.shuffle(self.Y_train)


    def gatherData(self):
        """
        TODO: FORWARD STEP
        """
        total_batch = self.train_df.shape[0] / (self.batch_size * self.seq_len)
        X_train = None

        print ('total_batch is: ', total_batch)
        for i in range(total_batch):
            train, tr_logerror, macro_features = self.next_batch()
            if X_train is None:
                X_train = train
                Y_train = tr_logerror
                macro_train = macro_features
            else:
                np.concatenate((X_train, train), axis=0)
                np.concatenate((Y_train, tr_logerror), axis=0)
                np.concatenate((macro_train, macro_features), axis=0)
                np.concatenate((macro_train, macro_features), axis=0)
                np.concatenate((macro_train, macro_features), axis=0)


        valid, valid_logerror, macro_valid = self.valid_data()

        test, test_logerror, macro_test = self.test_data()

        return X_train, Y_train, macro_train, valid, valid_logerror, macro_valid, test, test_logerror, macro_test

if __name__ == '__main__':
    d = Dataset()
    d.split(28000, 100, 100, 0)
    a = d.next_batch(3, 4)
    print(a)
    d.valid_data()
