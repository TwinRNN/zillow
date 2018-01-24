# encoding:utf-8
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import dataset_zillow
from PointProcessRNN_zillow import PointProcessRNN
import numpy as np


class BasicTrain(object):
    """
    # Adujst the logging level of tensorflow to Info
    tf.logging.set_verbosity(tf.logging.INFO)
    """

    def __init__(self, network_params, model_params, timeseries_params, best_valid_error, test_error):
        """
        Initialize a RNN with given parameters
        """


        self.MAX_TRAINING_STEP = model_params['max_step']
        """
        parameters for networks
        """
        self.network_params = network_params
        self.batch_size = network_params['batch_size']
        self.seq_len = network_params['seq_len']
        """
        parameters for models evolving over time
        """
        self.train_init = model_params['train_init']
        self.model_num = model_params['model_num']
        # self.model_step = model_params['model_step']
        self.valid_num = model_params['valid_num']
        self.test_num = model_params['test_num']
        self.forward = model_params['forward']
        """
        parameters for timeseries data
        """
        self.time_series_params = timeseries_params
        # self.time_series = timeseries_params['time_series']
        # self.delay = timeseries_params['delay']
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
        self.dataset = dataset_zillow.Dataset(self.batch_size, self.seq_len, self.time_series_params)
        """
        TODO: model_step
        """
        self.dataset.split(self.forward, self.train_init, self.valid_num,
                           self.test_num, self.model_num)
        tf.reset_default_graph()

        model = PointProcessRNN(self.network_params, self.time_series_step,
                                self.dataset.feature_size())
        graph = tf.Graph()
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=1.0, allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, gpu_options=gpu_options))
        # saver = tf.train.Saver()
        best_valid_error_local = 10
        valid_error_list = []
        house_valid, logerror_valid, macro_features_days_valid,macro_features_weeks_valid,macro_features_months_valid = self.dataset.valid_data()
        house_test, logerror_test, macro_features_days_test,macro_features_weeks_test,macro_features_months_test = self.dataset.test_data()

        with graph.as_default(), sess:

            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter("log/train")
            """
            TODO: EARLY STOPPING
            """
            for step in range(self.MAX_TRAINING_STEP):
                house_in, logerror_in, macro_features_days,macro_features_weeks,macro_features_months = self.dataset.next_batch()
                train_loss, train_error, train_summary = model.train_one_step(
                    sess, house_in, logerror_in, macro_features_days,macro_features_weeks,macro_features_months)
                train_writer.add_summary(train_summary, step)

                if step % 10 == 0:
                    print('step {}: training loss {};  error {};'.format(
                        step, train_loss, train_error))
                    # validate
                    
                    valid_loss, valid_error, test_summary = model.evaluate(
                        sess, house_valid, logerror_valid, macro_features_days_valid,macro_features_weeks_valid,macro_features_months_valid)
                    print('valid error{}'.format(valid_error))
                    valid_error_list.append(valid_error)
                    if valid_error < best_valid_error_local:
                        best_valid_error_local = valid_error
                        if valid_error < self.best_valid_error_global:
                            better_param = True
                            # saver.save(sess, "log/best.ckpt")
                            self.best_valid_error_global = valid_error
                            # If the global optimal validation error is
                            # updated, test the result
                            test_writer = tf.summary.FileWriter("log/test")
                            
                            test_loss, test_error, test_summary = model.evaluate(
                                sess, house_test, logerror_test, macro_features_days_test,macro_features_weeks_test,macro_features_months_test)
                            self.test_error_global = test_error

                    if step > 100 and abs(best_valid_error_local - valid_error) > 0.10:
                        break
                    if step > 500:
                        if abs(valid_error_list[-1] - valid_error_list[-2]) < 0.0002:
                            break
            # saver.restore(sess, "log/best.ckpt")
            print('best_valid_error_local: ', best_valid_error_local)
            print('best_valid_error_global:', self.best_valid_error_global)
            print('current_test_error_global: ', self.test_error_global)
            graph.finalize()
            sess.close()
            del sess
            del graph
            return best_valid_error_local, self.best_valid_error_global, self.test_error_global, better_param
