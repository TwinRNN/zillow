from keras.callbacks import EarlyStopping, Callback
from keras.layers import Dense, Input, LSTM, concatenate
from keras.layers.core import Reshape
from keras.models import Model
from keras.utils import multi_gpu_model
import dataset_zillow
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.optimizers import SGD, RMSprop, Adadelta, Adam


class BasicTrain(object):
    def __init__(self, network_params, model_params, timeseries_params, best_valid_error, test_error):
        """
        Initialize a RNN with given parameters
        """

        self.MAX_EPOCH = model_params['max_epoch']
        """
        parameters for networks
        """
        self.network_params = network_params
        self.batch_size = network_params['batch_size']
        self.seq_len = network_params['seq_len']
        self.state_size = network_params['state_size']
        self.learning_rate = network_params['lr']
        self.keep = network_params['pkeep']
        self.optimizer = network_params['optimizer']
        self.decay_rate = network_params['dr']
        self.activation_f = network_params['activation_f']
        self.initializer = network_params['initializer']
        """
        parameters for models evolving over time
        """
        self.train_init = model_params['train_init']
        self.model_num = model_params['model_num']
        self.valid_num = model_params['valid_num']
        self.test_num = model_params['test_num']
        self.forward = model_params['forward']
        """
        parameters for timeseries data
        """
        self.time_series_params = timeseries_params
        self.time_series_step = timeseries_params['time_series_step']

        print('-------------------------------------------------------------')
        print('network_params: ', network_params)
        print('model_paramas: ', model_params)
        print('timeseries_params:', timeseries_params)

        # to record the best error for
        self.best_valid_error_global = best_valid_error
        self.test_error_global = test_error

    def build(self):
        better_param = False
        dataset = dataset_zillow.Dataset(self.batch_size, self.seq_len, self.time_series_params)
        X_train, Y_train, macro_train_days, X_valid, Y_valid, macro_valid_days, X_test, Y_test, macro_test_days = dataset.gatherData()
        feature_size = dataset.feature_size()
        self.event_feature_size = feature_size['event']
        self.timeseries_feature_size = feature_size['time']

        opt = {"0": SGD(lr=self.learning_rate, decay=self.decay_rate),
               "1": RMSprop(lr=self.learning_rate, decay=self.decay_rate),
               "2": Adadelta(lr=self.learning_rate, decay=self.decay_rate),
               "3": Adam(lr=self.learning_rate, decay=self.decay_rate)}
        optimizer = opt["%d" % self.optimizer]

        history = self.LossHistory()
        callback = [EarlyStopping(monitor=self.cus_loss, patience=10, mode='min'), history]

        model = self.build_model()
        model2 = self.parallel_model(model)
        model2.compile(loss=self.cus_loss, optimizer=optimizer)
        model2.fit([X_train, macro_train_days], Y_train,
                   validation_data=([X_valid, macro_valid_days], Y_valid),
                   batch_size=self.batch_size, epoch=self.MAX_EPOCH, callbacks=callback, shuffle=False)

        valid_error = history.losses[-1]
        if valid_error < self.best_valid_error_global:
            better_param = True
            self.best_valid_error_global = valid_error
            y = model2.predict([X_test, macro_test_days])
            acc = np.mean(np.abs(1 - np.exp(Y_test - y)))
            self.test_error_global = acc

        print 'best_valid_error_global:', self.best_valid_error_global
        print 'current_test_error_global: ', self.test_error_global
        return valid_error, self.best_valid_error_global, self.test_error_global, better_param


    def build_model(self):
        with tf.device('/cpu:0'):
            # build house model
            house_input = Input(shape=(None, None, self.event_feature_size), dtype='float32')
            house_middle = LSTM(self.state_size, return_sequences=True, activation=self.activation_f,
                                kernel_initializer=self.initializer, dropout=self.keep)(house_input)
            house_output = LSTM(self.state_size, return_sequences=True, activation=self.activation_f,
                                kernel_initializer=self.initializer, dropout=self.keep)(house_middle)

            # build time model
            macro_input_days = Input(shape=(None, None, self.timeseries_feature_size), dtype='float32')
            macro_middle_days = LSTM(self.state_size, return_sequences=True, activation=self.activation_f,
                                     kernel_initializer=self.initializer, dropout=self.keep)(macro_input_days)
            macro_output_days = LSTM(self.state_size, return_sequences=False, activation=self.activation_f,
                                     kernel_initializer=self.initializer, dropout=self.keep)(macro_middle_days)
            macro_output_days = Reshape([self.batch_size, self.seq_len, self.state_size])(macro_output_days)


            # concatenate
            middle_output = concatenate([house_output, macro_output_days])
            finale_output = Dense(1, activation=self.activation_f)(middle_output)
            model = Model(inputs=[house_input, macro_input_days], outputs=finale_output)
        return model

    def parallel_model(self, model):
        parallel_model = multi_gpu_model(model, gpus=2)
        return parallel_model

    def cus_loss(self, y_true, y_pred):
        price_error = K.abs(1 - K.exp(y_true - y_pred))
        loss = K.mean(price_error)
        return loss

    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('val_loss'))


