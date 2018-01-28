# encoding:utf-8
import Train_parallel
import tensorflow as tf
import os
import argparse
from scipy.stats import bernoulli
from bitstring import BitArray
from deap import base, creator, tools, algorithms
import numpy as np
from keras.initializers import zeros, TruncatedNormal, Orthogonal, RandomUniform
import pickle


class Genetic(object):

    def __init__(self, population, generation, model_param, event_data, macro_days, macro_weeks, macro_months):
        """
        :param count:  count (int): Number of networks to generate, aka the size of the population
        """
        self.population = population
        self.generation = generation
        self.model_param = model_param

        """
        LR: 2^4  -> 4
        Rest network params: 2^2 -> 2 * 8
        Delay params: 2^3 -> 3 * 4 * 3
        time_series step: 2^2  -> 2 * 3
        4 + 16 + 36 + 6= 62

        """
        GENE_LENGTH = 62


        # one-variable minimize
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        # define parameter set
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register('binary', bernoulli.rvs, 0.5)
        self.toolbox.register('individual', tools.initRepeat, creator.Individual, self.toolbox.binary,
                              n=GENE_LENGTH)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

        # Genetic Operator
        self.toolbox.register('mate', tools.cxOrdered)
        self.toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.6)
        self.toolbox.register('select', tools.selRoulette)
        self.toolbox.register('evaluate', self.eval)

        self.best_valid_error_global = 10
        self.test_error = 0
        self.best_params = {}

    def eval(self, individual):
        """
        Evaluate Function
        :param individual:
        :return:
        """
        # network_param = {'batch_size': individual[0], 'seq_len':
        # individual[1], 'state_size': individual[2], ''}
        network_params, timeseries_params = self.decoder(individual)
        net = Train_parallel.BasicTrain(
            network_params, self.model_param, timeseries_params,
            self.best_valid_error_global, self.test_error)

        best_local, best_global, test_error, better_param = net.build(event_data, macro_days, macro_weeks, macro_months)
        self.best_valid_error_global = best_global
        self.test_error = test_error
        if better_param:
            self.best_params = {'network_params': network_params, 'timeseries_params': timeseries_params}
        return best_local,

    def search(self):
        population = self.toolbox.population(n=self.population)
        r = algorithms.eaSimple(population, self.toolbox, cxpb=0.4, mutpb=0.1,
                                ngen=self.generation, verbose=True)
        # best_individuals = tools.selBest(population, 1)[0]
        # all_individuals = tools.selBest(population, 2)
        # print ('pop_1:', all_individuals[0])
        # print ('pop_2:', all_individuals[1])
        # best_param = self.decoder(best_individuals)
        # print('best_params', best_param)
        # print('best_valid_error_via_deap', best_individuals.fitness.values)
        print('best_params: ', self.best_params)
        print('best_valid_error_via_record: ', self.best_valid_error_global)
        print('test_error: ', self.test_error)
        return self.best_params, self.best_valid_error_global, self.test_error

    def decoder(self, params):
        print('genetic_params:', params)
        NUM_NET_1 = 1  # for LR
        NUM_NET_2 = 8  # for the rest network params
        NUM_TIME = 3
        NUM_DELAY_TYPE = 3
        NUM_DELAY = 4

        # netowr_params
        BATCH_SIZE = [16, 32, 64, 128]
        SEQ_LEN = [16, 32, 64, 128]
        STATE_SIZE = [16, 32, 64, 128]
        LR = list(np.logspace(-3, -6, 16))
        DR = [0.99, 0.98, 0.97, 0.96]
        PKEEP = [0.9, 0.8, 0.7, 0.6]
        ACTIVATION = ["relu", "tanh", "sigmoid", "softsign"]
        INIT = [zeros(), TruncatedNormal(), Orthogonal(), RandomUniform()]
        net_name = ['lr', 'batch_size', 'seq_len', 'state_size', 'dr', 'pkeep', 'optimizer', 'activation_f', 'initializer']
        network_params = {}
        network_params['lr'] = LR[BitArray(params[0: NUM_NET_1 * 4]).uint]
        for i in range(NUM_NET_2):
            name = net_name[i + 1]
            network_params[name] = BitArray(params[4 + i * 2: 4 + i * 2 + 2]).uint
        network_params['batch_size'] = BATCH_SIZE[network_params['batch_size']]
        network_params['seq_len'] = SEQ_LEN[network_params['seq_len']]
        network_params['state_size'] = STATE_SIZE[network_params['state_size']]
        network_params['dr'] = DR[network_params['dr']]
        network_params['pkeep'] = PKEEP[network_params['pkeep']]
        network_params['activation_f'] = ACTIVATION[network_params['activation_f']]
        network_params['initializer'] = INIT[network_params['initializer']]

        # timeseries_params
        timeseries_params = {}

        TIME_STEP_DAYS = [7, 14, 30, 60]
        TIME_STEP_WEEKS = [4, 8, 12, 24]
        TIME_STEP_MONTHS = [2, 3, 6, 9]
        TIME_STEP = [TIME_STEP_DAYS, TIME_STEP_WEEKS, TIME_STEP_MONTHS]
        step_name = ['time_series_step_days', 'time_series_step_weeks', 'time_series_step_months']
        for index in range(NUM_TIME):
            name = step_name[index]
            step = TIME_STEP[index]
            timeseries_params[name] = step[BitArray(params[20 + index * 2: 20 + index * 2 + 2]).uint]

        DELAY = [7, 14, 30, 60, 90, 120, 150, 180]
        delay_name_days = ['delay_google_days', 'delay_tweeter_days', 'delay_macro_days', 'delay_tweeter_re_days']
        delay_name_weeks = ['delay_google_weeks', 'delay_tweeter_weeks', 'delay_macro_weeks', 'delay_tweeter_re_weeks']
        delay_name_months = ['delay_google_months', 'delay_tweeter_months', 'delay_macro_months', 'delay_tweeter_re_months']
        delay_name = [delay_name_days, delay_name_weeks, delay_name_months]
        for type in range(NUM_DELAY_TYPE):
            name_list = delay_name[type]
            for index in range(NUM_DELAY):
                name = name_list[index]
                timeseries_params[name] = DELAY[BitArray(params[26 + index * 3: 26 + index * 3 + 3]).uint]
        return network_params, timeseries_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    parser.add_argument('--model_index', type=str,
                        default='0', help='model index')
    parser.add_argument('--forward', type=int,
                        default=2560, help='number of steps for moving forward')
    parser.add_argument('--max_epoch', type=int,
                        default=1000, help='max number of epochs')
    parser.add_argument('--population', type=int,
                        default=30, help='population')
    parser.add_argument('--generation', type=int,
                        default=10, help='generation')
    parser.add_argument('--train_init', type=int,
                        default=128000, help='initial training points')
    parser.add_argument('--valid_num', type=int,
                        default=2560, help='validation points')
    parser.add_argument('--test_num', type=int, default=2560, help='test points')
    parser.add_argument('--buckets', type=str,
                            default='Data', help='input data path')
    # parser.add_argument('--time_series', type=bool, default=False, help='whether use times series data or not')
    # parser.add_argument('--delay_google', type=int, default=0, help='leading dates of google')
    # parser.add_argument('--delay_tweeter', type=int, default=0, help='leading dates of tweeter')
    # parser.add_argument('--delay_macro', type=int, default=0, help='leading dates of macro data')
    # parser.add_argument('--time_series_step', type=int, default=30, help='length of background information')
    FLAGS, _ = parser.parse_known_args()

    model_params = {'train_init': FLAGS.train_init, 'model_num': FLAGS.model_index, 'valid_num': FLAGS.valid_num,
                    'test_num': FLAGS.test_num, 'max_epoch': FLAGS.max_epoch, 'forward': FLAGS.forward}

    # Read data
    train_file_path = os.path.join(FLAGS.buckets, "zillow-model-data-original")
    train_file_path2 = os.path.join(FLAGS.buckets, "model_time_day")
    train_file_path3 = os.path.join(FLAGS.buckets, "model_time_week")
    train_file_path4 = os.path.join(FLAGS.buckets, "model_time_month")

    with tf.gfile.Open(train_file_path, 'rb') as f:
        raw_data = f.read()
        event_data = pickle.loads(raw_data)

    with tf.gfile.Open(train_file_path2, 'rb') as f:
        raw_data = f.read()
        data = pickle.loads(raw_data)
        macro_days = data['macro']

    with tf.gfile.Open(train_file_path3, 'rb') as f:
        raw_data = f.read()
        data = pickle.loads(raw_data)
        macro_weeks = data['macro']

    with tf.gfile.Open(train_file_path4,'rb') as f:
        raw_data = f.read()
        data = pickle.loads(raw_data)
        macro_months = data['macro']
    # best_params = []
    # errors = []
    # output= []

    model_index = FLAGS.model_index.split(',')
    for index in model_index:
        print('model: ', index)
        model_params['model_num'] = int(index)
        G = Genetic(FLAGS.population, FLAGS.generation, model_params, event_data, macro_days, macro_weeks, macro_months)
        best_ind, valid_err, test_error = G.search()
        file_name = 'output_model_%s_pop_%d_gen_%d.txt' % (
            index, FLAGS.population, FLAGS.generation)
        output_path = os.path.join(FLAGS.checkpointDir, file_name)
        print('output_path:', output_path)
        with tf.gfile.Open(output_path, 'wb') as wf:
            wf.write('model_num: ')
            wf.write(str(index) + '\n')
            wf.write('model_params: ')
            wf.write(str(model_params) + '\n')
            wf.write('best_network_params:')
            wf.write(str(best_ind['network_params']) + '\n')
            wf.write('best_time_series_params:')
            wf.write(str(best_ind['timeseries_params']) + '\n')
            wf.write('best_valid_error:')
            wf.write(str(valid_err) + '\n')
            wf.write('test_error:')
            wf.write(str(test_error) + '\n')
