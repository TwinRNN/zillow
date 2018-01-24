# encoding:utf-8
import numpy as np
import tensorflow as tf


class PointProcessRNN(object):

    def __init__(self, network_params, time_series_step_days,time_series_step_weeks,time_series_step_months, feature_size):
        """BATCH_SIZE = [16, 32, 64, 128]
        SEQ_LEN = [16, 32, 64, 128]
        STATE_SIZE = [16, 32, 64, 128]
        LR = list(np.logspace(-3, -6, 16))
        DR = [0.99, 0.98, 0.97, 0.96]
        PKEEP = [0.9, 0.8, 0.7, 0.6]
        ACTIVATION = [tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid, tf.nn.softsign]
        INIT = [tf.truncated_normal_initializer(stddev=0.01), tf.random_uniform_initializer(), tf.zeros_initializer(),
                tf.orthogonal_initializer()]"""

        self.state_size = network_params['state_size']
        self.learning_rate = network_params['lr']
        self.keep = network_params['pkeep']
        self.optimizer = network_params['optimizer']
        self.decay_rate = network_params['dr']
        self.activation_f = network_params['activation_f']
        self.seq_len = network_params['seq_len']
        self.batch_size = network_params['batch_size']
        self.initializer = network_params['initializer']

        self.losses = []
        self.errors = []
        self.step = 0

        self.event_feature_size = feature_size['event']
        self.timeseries_feature_size = feature_size['time']
        self.num_step_timeseries_days = time_series_step_days
        self.num_step_timeseries_weeks = time_series_step_weeks
        self.num_step_timeseries_months = time_series_step_months
        self.build()

    def build(self):
        # drop out时输出被保留的概率



        self.pkeep = tf.placeholder(
            tf.float32, name='pkeep')  # dropout parameter

        with tf.name_scope('inputs'):
            self.house_ph = tf.placeholder(
                tf.float32, [None, None, self.event_feature_size], name='house_ph')
            self.logerror = tf.placeholder(
                tf.float32, [None, None, 1], name='logerror')
            self.time_series_days = tf.placeholder(
                tf.float32, [None, None, None, self.timeseries_feature_size], name='time_series')
            self.time_series_weeks = tf.placeholder(
                tf.float32, [None, None, None, self.timeseries_feature_size], name='time_series')
            self.time_series_months = tf.placeholder(
                tf.float32, [None, None, None, self.timeseries_feature_size], name='time_series')
        batch_size = tf.shape(self.house_ph)[0]
        seq_len = tf.shape(self.house_ph)[1]


        with tf.variable_scope("house"):
            # cell = tf.contrib.rnn.LSTMCell(self.state_size, state_is_tuple=True)
            cells = [tf.contrib.rnn.GRUCell(self.state_size, activation=self.activation_f,
                                            kernel_initializer=self.initializer,
                                            bias_initializer=tf.zeros_initializer()) for _ in range(2)]
            cells = [tf.contrib.rnn.DropoutWrapper(
                cell, input_keep_prob=self.pkeep) for cell in cells]
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=False)
            # cell = tf.contrib.rnn.AttentionCellWrapper(cell, self.state_size)
            # init_state = tf.random_normal([batch_size, self.state_size])
            # init_state = cell.zero_state(batch_size, tf.float32)
            rnn_house_outputs, H_house = tf.nn.dynamic_rnn(
                cell, self.house_ph, dtype=tf.float32)


        with tf.variable_scope("time_series"):
            cells = [tf.contrib.rnn.GRUCell(self.state_size, activation=self.activation_f,
                                            kernel_initializer=self.initializer,
                                            bias_initializer=tf.zeros_initializer()
                                            ) for _ in range(2)]
            cells = [tf.contrib.rnn.DropoutWrapper(
                cell, input_keep_prob=self.pkeep) for cell in cells]
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=False)
            time_series_days = tf.reshape(
                self.time_series_days, [-1, self.num_step_timeseries_days, self.timeseries_feature_size])
            # init_state = cell.zero_state(batch_size * seq_len, tf.float32)
            rnn_time_outputs_days, _ = tf.nn.dynamic_rnn(
                cell, time_series_days, dtype=tf.float32)
            rnn_time_outputs_days = tf.reshape(
                rnn_time_outputs_days[:, -1, :], [batch_size, seq_len, self.state_size])
        with tf.variable_scope("time_series"):
            cells = [tf.contrib.rnn.GRUCell(self.state_size, activation=self.activation_f,
                                            kernel_initializer=self.initializer,
                                            bias_initializer=tf.zeros_initializer()
                                            ) for _ in range(2)]
            cells = [tf.contrib.rnn.DropoutWrapper(
                cell, input_keep_prob=self.pkeep) for cell in cells]
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=False)
            time_series_weeks = tf.reshape(
                self.time_series_weeks, [-1, self.num_step_timeseries_weeks, self.timeseries_feature_size])
            # init_state = cell.zero_state(batch_size * seq_len, tf.float32)
            rnn_time_outputs_weeks, _ = tf.nn.dynamic_rnn(
                cell, time_series_weeks, dtype=tf.float32)
            rnn_time_outputs_weeks = tf.reshape(
                rnn_time_outputs_weeks[:, -1, :], [batch_size, seq_len, self.state_size])
        with tf.variable_scope("time_series"):
            cells = [tf.contrib.rnn.GRUCell(self.state_size, activation=self.activation_f,
                                            kernel_initializer=self.initializer,
                                            bias_initializer=tf.zeros_initializer()
                                            ) for _ in range(2)]
            cells = [tf.contrib.rnn.DropoutWrapper(
                cell, input_keep_prob=self.pkeep) for cell in cells]
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=False)
            time_series_months = tf.reshape(
                self.time_series_months, [-1, self.num_step_timeseries_months, self.timeseries_feature_size])
            # init_state = cell.zero_state(batch_size * seq_len, tf.float32)
            rnn_time_outputs_months, _ = tf.nn.dynamic_rnn(
                cell, time_series_months, dtype=tf.float32)
            rnn_time_outputs_months = tf.reshape(
                rnn_time_outputs_months[:, -1, :], [batch_size, seq_len, self.state_size])


        with tf.name_scope("concat"):
            concat_outputs = tf.concat(
                [rnn_house_outputs, rnn_time_outputs_days, rnn_time_outputs_weeks, rnn_time_outputs_months], axis=2)

        with tf.name_scope("prediction"):
            prediction = tf.contrib.layers.linear(
                concat_outputs, 1, activation_fn=self.activation_f)
            prediction = tf.identity(prediction, name='prediction')

        with tf.name_scope("error"):
            price_error = tf.abs(1 - tf.exp(logerror-prediction))
            self.price_error = tf.reduce_mean(price_error, name='error')
            tf.summary.scalar('price error', self.price_error)

        # MSE loss
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(logerror - prediction))
            tf.summary.scalar('loss', self.loss)


    """
    def train_one_step(self, sess, house_in, logerror_in, time_series_days, time_series_weeks, time_series_months):
        feed_dict = {self.house_ph: house_in, self.logerror: logerror_in, self.time_series_days: time_series_days,
                     self.time_series_weeks: time_series_weeks, self.time_series_months: time_series_months, self.pkeep: self.keep}

        loss, error, summary, _ = sess.run(
            [self.loss, self.price_error, self.merged, self.train_op], feed_dict=feed_dict)
        sess.run(self.update_step)

        self.step += 1
        self.losses.append(loss)
        self.errors.append(error)

        return loss, error, summary
    """
    """
    def evaluate(self, sess, house_in, logerror_in, time_series_days, time_series_weeks, time_series_months):
        feed_dict = {self.house_ph: house_in, self.logerror: logerror_in, self.time_series_days: time_series_days,
                     self.time_series_weeks: time_series_weeks, self.time_series_months: time_series_months, self.pkeep: 1.0}

        loss, error, summary = sess.run(
            [self.loss, self.price_error, self.merged], feed_dict=feed_dict)

        return loss, error, summary
    """
