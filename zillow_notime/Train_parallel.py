from keras.callbacks import EarlyStopping, Callback
from keras.layers import Dense, LSTM, RNN, StackedRNNCells, Input
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
import numpy as np
import tensorflow as tf
import pickle
from keras import backend as K
import argparse
import os

class Train(object):

    def __init__(self, network_params, model_params, best_valid_error, test_error):
        """
        Initialize a RNN with given parameters
        """
        self.MAX_TRAINING_STEP = model_params['max_step']
        """
        parameters for networks
        """
        self.network_params = network_params
        self.state_size = network_params['state_size']
        self.learning_rate = network_params['lr']
        self.keep = network_params['pkeep']
        self.optimizer = network_params['optimizer']
        self.decay_rate = network_params['dr']
        self.activation_f = network_params['activation_f']
        self.seq_len = network_params['seq_len']
        self.batch_size = network_params['batch_size']
        self.initializer = network_params['initializer']
        """
        parameters for models evolving over time
        """
        self.train_init = model_params['train_init']
        self.model_num = model_params['model_num']
        # self.model_step = model_params['model_step']
        self.valid_num = model_params['valid_num']
        self.test_num = model_params['test_num']
        self.forward = model_params['forward']
        print ('-------------------------------------------------------------')
        print('network_params: ', network_params)
        print('model_params: ', model_params)
        # to record the best error for
        self.best_valid_error_global = best_valid_error
        self.test_error_global = test_error
        self.dataset()

    def train(self):

        better_param = False

        history = self.LossHistory()
        callback = [EarlyStopping(monitor='val_loss', patience=5, mode='min'), history]
        X_train = np.reshape(self.train, [-1, self.seq_len, self.feature_size])
        '''
        TODO: ??? VAL SHAPE
        '''
        X_val = np.reshape(self.valid, [1, -1, self.feature_size])
        Y_train = np.reshape(self.tr_logerror, [-1, self.seq_len, 1])
        Y_val = np.reshape(self.valid_logerror, [1, -1, 1])
        # print(X_train.shape)
        # print(Y_train.shape)
        model = self.build_model()

        model2 = self.parallel_model(model)
        #build the map of opt
        opt = {"1": SGD(lr=self.learning_rate, decay=self.decay_rate),
                "2": RMSprop(lr=self.learning_rate, decay=self.decay_rate),
                "3": Adadelta(lr=self.learning_rate, decay=self.decay_rate),
                "4": Adam(lr=self.learning_rate, decay=self.decay_rate)}
        optimizer = opt["%d" % self.optimizer]

        model2.compile(loss=self.cus_loss, optimizer=optimizer)

        model2.fit(X_train, Y_train, epochs=self.MAX_TRAINING_STEP, batch_size=self.batch_size,
                   callbacks=[callback, ], validation_data=(X_val, Y_val), shuffle=False)

        valid_error = history.losses[-1]
        if valid_error < self.best_valid_error_global:
            better_param = True
            self.best_valid_error_global = valid_error
            X_test = self.test.reshape(1, -1, self.feature_size)
            Y_test = self.test_logerror.reshape((1, -1, 1))
            y = model2.predict(X_test)
            acc = np.mean(np.abs(1 - np.exp(Y_test - y)))
            self.test_error_global = acc

        print 'best_valid_error_global:', self.best_valid_error_global
        print 'current_test_error_global: ', self.test_error_global
        return valid_error, self.best_valid_error_global, self.test_error_global, better_param

    def build_model(self):
        with tf.device('/cpu:0'):
            '''
            #this kind of model will be abandoned
            model = Sequential()
            cells = [LSTMCell(self.state_size) for _ in range(2)]
            cell = StackedRNNCells(cells)
            model.add(RNN(cell, return_sequences=True, input_shape=[None, self.feature_size]))
            model.add(Dense(1))
            '''
            #add the activation Function
            #add initializer
            #add dropout layer
            house_input = Input(shape=(None, None, self.feature_size), dtype='float32')
            house_middle = LSTM(self.state_size, return_sequences=True, activation=self.activation_f,
                                kernel_initializer=self.initializer, dropout=self.keep)(house_input)
            house_output = LSTM(self.state_size, return_sequences=True, activation=self.activation_f,
                                kernel_initializer=self.initializer, dropout=self.keep)(house_middle)
            finale_output = Dense(1,activation=self.activation_f)(house_output)
            model = Model(inputs=house_input, outputs=finale_output)
            return model

    def parallel_model(self, model):
        parallel_model = multi_gpu_model(model, gpus=2)
        return parallel_model

    def cus_loss(self, y_true, y_pred):
        price_error = K.abs(1 - K.exp(y_true - y_pred))
        loss = K.mean(price_error)
        return loss

    def dataset(self):
        parser = argparse.ArgumentParser()

        # parser.add_argument('--buckets', type=str,
                            # default='', help='input data path')

        # FLAGS, _ = parser.parse_known_args()
        # train_file_path = os.path.join(FLAGS.buckets, "zillow-model-data-original")

        train_file_path = "data/zillow-model-data-original"

        with tf.gfile.Open(train_file_path, 'rb') as f:
            raw_data = f.read()
            data = pickle.loads(raw_data)

        train_df = data['train_df']
        logerror_df = data['logerror_df']

        train_end = self.train_init + self.model_num * self.forward
        self.train = train_df[: train_end].values
        self.tr_logerror = logerror_df[: train_end].values

        valid_end = train_end + self.valid_num
        self.valid = train_df[train_end: valid_end].values
        self.valid_logerror = logerror_df[train_end: valid_end].values

        test_end = valid_end + self.test_num
        self.test = train_df[valid_end: test_end].values
        self.test_logerror = logerror_df[valid_end: test_end].values

        self.feature_size = train_df.shape[1]

    class LossHistory(Callback):

        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('val_loss'))
