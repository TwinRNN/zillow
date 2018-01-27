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
                            default='', help='input data path')
        # 配置模型保存地址
        parser.add_argument('--output_dir', type=str,
                            default='', help='output model path')

        FLAGS, _ = parser.parse_known_args()
        train_file_path = os.path.join(FLAGS.buckets, "zillow-model-data-original")
        train_file_path2 = os.path.join(FLAGS.buckets, "model_time_day")
        train_file_path3 = os.path.join(FLAGS.buckets, "model_time_week")
        train_file_path4 = os.path.join(FLAGS.buckets, "model_time_month")

        with tf.gfile.Open(train_file_path, 'rb') as f:
            raw_data = f.read()
            data = pickle.loads(raw_data)
            self.train_df = data['train_df'][: 150000]
            self.logerror_df = data['logerror_df'][: 150000]
            self.transactiondate_df = data['transactiondate_df'][: 150000]

        with tf.gfile.Open(train_file_path2, 'rb') as f:
            raw_data = f.read()
            data = pickle.loads(raw_data)
            self.macro_days = data['macro']
            self.google_days = self.macro_days[:, : 15]
            self.economy_days = self.macro_days[:, 15: 23]
            self.tweet_days = self.macro_days[:, 23: 29]
            self.tweet_re_days = self.macro_days[:, 29: 35]

        with tf.gfile.Open(train_file_path3, 'rb') as f:
            raw_data = f.read()
            data = pickle.loads(raw_data)
            self.macro_weeks = data['macro']
            self.google_weeks = self.macro_weeks[:, : 15]
            self.economy_weeks = self.macro_weeks[:, 15: 23]
            self.tweet_weeks = self.macro_weeks[:, 23: 29]
            self.tweet_re_weeks = self.macro_weeks[:, 29: 35]

        with tf.gfile.Open(train_file_path4,'rb') as f:
            raw_data = f.read()
            data = pickle.loads(raw_data)
            self.macro_months = data['macro']
            self.google_months = self.macro_months[:, : 15]
            self.economy_months = self.macro_months[:, 15: 23]
            self.tweet_months = self.macro_months[:, 23: 29]
            self.tweet_re_months = self.macro_months[:, 29: 35]

        self.batch_size = batch_size
        self.seq_len = seq_len

        self.time_series_params = time_series_params
        # three different kinds of time params
        self.time_series_step_days = time_series_params['time_series_step_days']
        self.time_series_step_weeks = time_series_params['time_series_step_weeks']
        self.time_series_step_months = time_series_params['time_series_step_months']

        delay_google_days = time_series_params['delay_google_days']
        delay_tweeter_days = time_series_params['delay_tweeter_days']
        delay_macro_days = time_series_params['delay_macro_days']
        delay_tweeter_re_days = time_series_params['delay_tweeter_re_days']

        delay_google_weeks = time_series_params['delay_google_weeks']
        delay_tweeter_weeks = time_series_params['delay_tweeter_weeks']
        delay_macro_weeks = time_series_params['delay_macro_weeks']
        delay_tweeter_re_weeks = time_series_params['delay_tweeter_re_weeks']

        delay_google_months = time_series_params['delay_google_months']
        delay_tweeter_months = time_series_params['delay_tweeter_months']
        delay_macro_months = time_series_params['delay_macro_months']
        delay_tweeter_re_months = time_series_params['delay_tweeter_re_months']

        self.move_google_days = MAP - self.time_series_step_days - delay_google_days
        self.move_tweeter_days = MAP - self.time_series_step_days - delay_tweeter_days
        self.move_macro_days = MAP - self.time_series_step_days - delay_macro_days
        self.move_tweeter_re_days = MAP - self.time_series_step_days - delay_tweeter_re_days

        self.move_google_weeks = MAP - self.time_series_step_weeks * 7- delay_google_weeks
        self.move_tweeter_weeks = MAP - self.time_series_step_weeks * 7 - delay_tweeter_weeks
        self.move_macro_weeks = MAP - self.time_series_step_weeks * 7 - delay_macro_weeks
        self.move_tweeter_re_weeks = MAP - self.time_series_step_weeks * 7 - delay_tweeter_re_weeks

        self.move_google_months = MAP - self.time_series_step_months * 30 - delay_google_months
        self.move_tweeter_months = MAP - self.time_series_step_months * 30- delay_tweeter_months
        self.move_macro_months = MAP - self.time_series_step_months * 30 - delay_macro_months
        self.move_tweeter_re_months = MAP - self.time_series_step_months * 30- delay_tweeter_re_months

        self.epochs = 0
        self.cursor = 0

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
            1], "time": self.macro_days.shape[1]}
        return feature_size

    def get_time_series(self, transactiondate, cursor, batch, seq_len,
                        timeGoogle, timeMacro, timeTweet, timeTweet_re,
                        timeSteps, moveGoogle, moveMacro, moveTweeter, moveTweeter_re, step):

        data_set = batch * seq_len
        google = []
        macro = []
        tweet = []
        tweet_re = []

        for i in range(data_set):
            init_time_google = transactiondate[i + cursor] + moveGoogle
            init_time_macro = transactiondate[i + cursor] + moveMacro
            init_time_tweet = transactiondate[i + cursor] + moveTweeter
            init_time_tweet_re = transactiondate[i + cursor] + moveTweeter_re

            for _ in range(timeSteps):
                google.append(timeGoogle[init_time_google + _ * step])
                macro.append(timeMacro[init_time_macro + _ * step])
                tweet.append(timeTweet[init_time_tweet + _ * step])
                tweet_re.append(timeTweet_re[init_time_tweet_re + _ * step])
        # print('google_shape: ', np.array(macro_features_google).shape)
        # print('tweeter_shape: ', np.array(macro_features_tweet).shape)
        # print('macro_shape: ', np.array(macro_features_macro).shape)
        macro_features = np.concatenate((tweet, tweet_re,
                                         macro, google), axis=1)
        macro_features = np.asarray(macro_features).reshape(
            [batch, seq_len, timeSteps, -1])
        return macro_features

    def next_batch(self):

        data_set = self.batch_size * self.seq_len
        if self.cursor + data_set > self.size:
            self.epochs += 1
            self.cursor = 0
            self.roll(self.batch_size, self.seq_len)

        train = self.train[self.cursor: self.cursor + data_set]
        tr_logerror = self.tr_logerror[self.cursor: self.cursor + data_set]

        macro_features_days = self.get_time_series(self.tr_transactiondate, self.cursor, self.batch_size,
                                                   self.seq_len, self.google_days, self.economy_days,
                                                   self.tweet_days, self.tweet_re_days, self.time_series_step_days,
                                                   self.move_google_days, self.move_macro_days, self.move_tweeter_days,
                                                   self.move_tweeter_re_days, step=1)
        macro_features_weeks = self.get_time_series(self.tr_transactiondate, self.cursor, self.batch_size,
                                                    self.seq_len, self.google_weeks, self.economy_weeks,
                                                    self.tweet_weeks, self.tweet_re_weeks, self.time_series_step_weeks,
                                                    self.move_google_weeks, self.move_macro_weeks, self.move_tweeter_weeks,
                                                    self.move_tweeter_re_weeks, step=7)
        macro_features_months = self.get_time_series(self.tr_transactiondate, self.cursor, self.batch_size,
                                                     self.seq_len, self.google_months, self.economy_months,
                                                     self.tweet_months, self.tweet_re_months,
                                                     self.time_series_step_months, self.move_google_months,
                                                     self.move_macro_months, self.move_tweeter_months,
                                                     self.move_tweeter_re_months, step=30)
        self.cursor += data_set

        train = train.reshape([self.batch_size, self.seq_len, -1])
        tr_logerror = tr_logerror.reshape([self.batch_size, self.seq_len, -1])

        return train, tr_logerror, macro_features_days, macro_features_weeks, macro_features_months

    def valid_data(self):
        valid = self.valid[np.newaxis]
        seqlen = valid.shape[1]
        valid_logerror = self.valid_logerror.reshape((1, -1, 1))

        macro_features_days = self.get_time_series(self.valid_transactiondate, 0, 1, seqlen, self.google_days,
                                                   self.economy_days, self.tweet_days, self.tweet_re_days,
                                                   self.time_series_step_days, self.move_google_days,
                                                   self.move_macro_days, self.move_tweeter_days,
                                                   self.move_tweeter_re_days, step=1)
        macro_features_weeks = self.get_time_series(self.valid_transactiondate, 0, 1, seqlen, self.google_weeks,
                                                    self.economy_weeks, self.tweet_weeks, self.tweet_re_weeks,
                                                    self.time_series_step_weeks, self.move_google_weeks,
                                                    self.move_macro_weeks,self.move_tweeter_weeks,self.move_tweeter_re_weeks, step=7)
        macro_features_months = self.get_time_series(self.valid_transactiondate, 0, 1, seqlen,
                                                     self.google_months, self.economy_months, self.tweet_months,
                                                     self.tweet_re_months, self.time_series_step_months,
                                                     self.move_google_months, self.move_macro_months,
                                                     self.move_tweeter_months, self.move_tweeter_re_months, step=30)
        return valid, valid_logerror, macro_features_days, macro_features_weeks, macro_features_months

    def test_data(self):
        test = self.test[np.newaxis]
        seqlen = test.shape[1]
        test_logerror = self.test_logerror.reshape((1, -1, 1))
        macro_features_days = self.get_time_series(self.test_transactiondate, 0, 1, seqlen, self.google_days,
                                                   self.economy_days, self.tweet_days, self.tweet_re_days,
                                                   self.time_series_step_days, self.move_google_days,
                                                   self.move_macro_days, self.move_tweeter_days,
                                                   self.move_tweeter_re_days, step=1)
        macro_features_weeks = self.get_time_series(self.test_transactiondate, 0, 1, seqlen, self.google_weeks,
                                                    self.economy_weeks, self.tweet_weeks, self.tweet_re_weeks,
                                                    self.time_series_step_weeks, self.move_google_weeks,
                                                    self.move_macro_weeks, self.move_tweeter_weeks,
                                                    self.move_tweeter_re_weeks, step=7)
        macro_features_months = self.get_time_series(self.test_transactiondate, 0, 1, seqlen, self.google_months,
                                                     self.economy_months, self.tweet_months, self.tweet_re_months,
                                                     self.time_series_step_months, self.move_google_months,
                                                     self.move_macro_months, self.move_tweeter_months,
                                                     self.move_tweeter_re_months, step=30)
        return test, test_logerror, macro_features_days, macro_features_weeks, macro_features_months

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
            train, tr_logerror, macro_features_days, macro_features_weeks, macro_features_months = self.next_batch()
            if X_train is None:
                X_train = train
                Y_train = tr_logerror
                macro_train_days = macro_features_days
                macro_train_months = macro_features_months
                macro_train_weeks = macro_features_weeks
            else:
                np.concatenate((X_train, train), axis=0)
                np.concatenate((Y_train, tr_logerror), axis=0)
                np.concatenate((macro_train_days, macro_features_days), axis=0)
                np.concatenate((macro_train_weeks, macro_features_weeks), axis=0)
                np.concatenate((macro_train_months, macro_features_months), axis=0)

        valid, valid_logerror, macro_valid_days, macro_valid_weeks, macro_valid_months = self.valid_data()

        test, test_logerror, macro_test_days, macro_test_weeks, macro_test_months = self.test_data()

        return X_train, Y_train, macro_train_days, macro_train_weeks, macro_train_months, \
               valid, valid_logerror, macro_valid_days, macro_valid_weeks, macro_valid_months, \
               test, test_logerror, macro_test_days, macro_test_weeks, macro_test_months

if __name__ == '__main__':
    d = Dataset()
    d.split(28000, 100, 100, 0)
    a = d.next_batch()
    print(a)
    d.valid_data()
