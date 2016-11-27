import matplotlib
matplotlib.use('Agg')
import pandas as pd
from functions import *


# Data class contains the data and extracts all the details.
class Data():
    def __init__(self, input_file, batch_size):
        # type: (object, object) -> object
        # type: (object, object) -> object
        """
        :param input_file: path to the input_file
        :param batch_size: size of the batches to use with this data
        """
        data = pd.read_table(input_file)
        data.aaMutations = data.aaMutations.fillna('')
        unique_mutations = set(':'.join(data.aaMutations).split(':'))
        unique_mutations.remove('')
        unique_mutations = sorted(list(unique_mutations))
        self.data = data
        self.unique_mutations = unique_mutations
        self.batch_size = batch_size
        self.input_file = input_file
        self.nn_genotypes_values, self.nn_brightness_values = format_data(data, unique_mutations)
        self.batches = get_batches(self.nn_genotypes_values, self.nn_brightness_values, batch_size, unique_mutations)
        self.batch_number = len(self.batches)
        self.to_plot_observed = self.nn_brightness_values[0:(self.batch_number * self.batch_size)]
        self.nn_genotypes = tf.placeholder(tf.float32, shape=[self.batch_size, 1, len(unique_mutations)])
        self.nn_brightness = tf.placeholder(tf.float32, shape=[self.batch_size, 1, 1])

    def reshuffle(self):
        self.nn_genotypes_values, self.nn_brightness_values = format_data(self.data, self.unique_mutations)
        self.batches = get_batches(self.nn_genotypes_values, self.nn_brightness_values, self.batch_size, self.unique_mutations)
        self.to_plot_observed = self.nn_brightness_values[0:(self.batch_number * self.batch_size)]


# Neural network class. Extracts neural net structure from the parameter file.
# Contains all the details of the neural network to be used.
class TFNet(object):
    def __init__(self, net_structure, input_data, optimizer_method, learning_rate, batch_size, cost_stats_file):
        '''
            :param net_structure:
                                {'layer1':(3, tf.tanh()),
                                'layer2':((3, tf.tanh()),
                                'layer3':(1, tf.tanh())}

            :return:

            https://www.tensorflow.org/versions/r0.9/api_docs/python/nn.html#activation-functions

            '''

        self.number_of_layers = len(net_structure)
        self.structure = net_structure

        self.neurons = {}
        self.weights = {}
        self.biases = {}
        self.input = {}
        self.output = {}

        for i in range(self.number_of_layers):
            layer = 'layer' + str(i + 1)
            self.neurons[layer] = int(self.structure[layer][0])
            self.weights[layer] = tf.Variable(
                tf.random_normal([1, len(input_data.unique_mutations), self.neurons[layer]]),
                name=layer + '_weights')
            self.biases[layer] = tf.Variable(tf.random_normal([1, 1, self.neurons[layer]]), name=layer + '_biases')
            self.input[layer] = tf.add(
                tf.batch_matmul(input_data.nn_genotypes, broadcast(self.weights[layer], batch_size)),
                broadcast(self.biases[layer], batch_size))
            self.output[layer] = eval(self.structure[layer][1])(self.input[layer])

        self.cost = tf.reduce_sum(tf.pow(self.output[layer] - input_data.nn_brightness, 2)) / batch_size
        self.optimizer = eval(optimizer_method)(learning_rate).minimize(self.cost)

        self.init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()

        self.cost_stats_file = cost_stats_file

    def __str__(self):
        print('Net structure:\n')
        for i in range(self.number_of_layers):
            print('%s neurons in layer_' % (self.neurons['layer' + str(i + 1)]) + str(i + 1) + '\n')
