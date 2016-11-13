import matplotlib
matplotlib.use('Agg')
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import argparse
import boto3

s3 = boto3.resource('s3')

parser = argparse.ArgumentParser(description='''

*** Only 3-layered nets are currently supported ***

Parameters file has to contain 8 parameters -- each on a new line.
Each line has to start with a parameter name, followed by tab and the parameter meaning.


The parameters are:

1) input_file (Path to the input file)

2) output_folder (Path to the output folder)

3) learning_rate (Learning rate)

4) batch_size (Batch size)

5) number_of_iterations (Number of iterations)

6) reshuffling_frequency (Once per how many iterations would you like to reshuffle your data?)

7) optimizer (Optimizer method,
please choose one from https://www.tensorflow.org/versions/r0.9/api_docs/python/train.html#optimizers)

8) net_structure (Net structure)


Net structure has to be formatted the following way:

net_structure	1,tf.tanh	3,tf.tanh	1,tf.tanh

Each layer of the network has to be separated by a tab.
The layer description consists of 2 parameters, separated by comma:

-- the number of neurons in the layer
-- the neuron output function

It is possible to choose from a variety of output functions, listed here:

https://www.tensorflow.org/versions/r0.9/api_docs/python/nn.html#activation-functions

!! It is very important to precisely follow the punctuation !!
Otherwise, the code will give an error.

''', formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('parameters_file')

arguments = parser.parse_args()
net_structure = {}

# Parsing the parameters document (details are in help above)
for line in open(arguments.parameters_file).readlines():
    if 'INPUT' in line.upper():
        input_file = str(line.rstrip('\n').split('\t')[1])
    if 'OUTPUT' in line.upper():
        output_folder = str(line.rstrip('\n').split('\t')[1])
    if 'LEARNING RATE' in line.upper() or 'LEARNING_RATE' in line.upper():
        learning_rate = float(line.rstrip('\n').split('\t')[1])
    if 'BATCH SIZE' in line.upper() or 'BATCH_SIZE' in line.upper():
        batch_size = int(line.rstrip('\n').split('\t')[1])
    if 'ITERATIONS' in line.upper():
        max_epochs = int(line.rstrip('\n').split('\t')[1])
    if 'RESHUFFLING' in line.upper():
        reshuffling_frequency = int(line.rstrip('\n').split('\t')[1])
    if 'OPTIMIZE' in line.upper():
        optimizer_method = str(line.rstrip('\n').split('\t')[1])
    if 'STRUCTURE' in line.upper():
        counter = 1
        for i in line.split('\t')[1:]:
            net_structure['layer' + str(counter)] = i.split(',')
            counter += 1

print('Folder name for this run is %s' % output_folder)
if output_folder[-1] != '/':
    output_folder += '/'

# Help dummy files that will be generated locally, before sending to s3. They'll be overwritten each iteration.
cost_stats_file = output_folder + 'cost_stats.txt'
temp_fig_file = output_folder + 'temp_fig.png'
temp_model_ckpt_file = output_folder + 'model.ckpt'

print('# learning rate = %s' % learning_rate)
print('# number of iterations = %s' % max_epochs)
print('# batch size = %s' % batch_size)
print('# reshuffling frequency = %s' % reshuffling_frequency)


# Functions for plotting. The first is to actually plot, the second is to make the plot readable.
def density_plot(x, y):
    ''' x = observed, y = predicted '''
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]
    y = y[mask]

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=10, edgecolor='')


def format_plot(ax, iteration_number, costs):
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('Observed brightness')
    plt.ylabel('Predicted brightness')
    plt.title('Iteration %s: cost=%.7f' % (iteration_number, costs))

    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color('gray')
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color('gray')
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    return ax


# Function for shaping the data. It performes reshuffling of the data,
# brightness normalization, and extracts genotype and brightness to separate matrices.
def format_data(data, unique_mutations):
    # shuffling rows in the data df
    data = data.reindex(np.random.permutation(data.index))
    print('Normalizing data...')
    # formatting data for the nn input
    nn_genotypes_values = np.zeros((len(data), len(unique_mutations)))
    nn_brightness_values = data.medianBrightness.values
    for i in range(len(unique_mutations)):
        nn_genotypes_values[:, i] = data.aaMutations.str.contains(unique_mutations[i]).astype(np.float32)

    nn_brightness_values = (nn_brightness_values - min(nn_brightness_values)) / max(
        nn_brightness_values - min(nn_brightness_values)) * 2 - 1
    return nn_genotypes_values, nn_brightness_values


# Function for generating batches from the data.
def get_batches(nn_genotypes_values, nn_brightness_values, batch_size, unique_mutations):
    nn_brightness_values_1 = nn_brightness_values
    print('Creating batches...')
    batches = []
    batch_number = int(nn_genotypes_values.shape[0] / batch_size)
    for i in range(batch_number):
        current_batch = nn_genotypes_values[batch_size * i:batch_size * (i + 1), :].reshape(batch_size, 1,
                                                                                            len(unique_mutations))
        current_batch_brightness = nn_brightness_values_1[batch_size * i:batch_size * (i + 1)].reshape(batch_size, 1, 1)
        batches.append((current_batch, current_batch_brightness))
    return batches


# Function for broadcasting a tensor (used before multiplication with weights).
def broadcast(tensor, batch_size):
    return tf.tile(tensor, (batch_size, 1, 1))


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
        self.unique_mutations = unique_mutations

        self.nn_genotypes_values, self.nn_brightness_values = format_data(data, unique_mutations)
        self.batches = get_batches(self.nn_genotypes_values, self.nn_brightness_values, batch_size, unique_mutations)
        self.batch_number = len(self.batches)
        self.to_plot_observed = self.nn_brightness_values[0:(self.batch_number * batch_size)]
        self.nn_genotypes = tf.placeholder(tf.float32, shape=[batch_size, 1, len(unique_mutations)])
        self.nn_brightness = tf.placeholder(tf.float32, shape=[batch_size, 1, 1])

    def reshuffle(self):
        data = pd.read_table(input_file)
        data.aaMutations = data.aaMutations.fillna('')
        unique_mutations = set(':'.join(data.aaMutations).split(':'))
        unique_mutations.remove('')
        unique_mutations = sorted(list(unique_mutations))
        self.unique_mutations = unique_mutations

        self.nn_genotypes_values, self.nn_brightness_values = format_data(data, unique_mutations)
        self.batches = get_batches(self.nn_genotypes_values, self.nn_brightness_values, batch_size, unique_mutations)
        self.to_plot_observed = self.nn_brightness_values[0:(self.batch_number * batch_size)]


# Neural network class. Extracts neural net structure from the parameter file.
# Contains all the details of the neural network to be used.
class TFNet(object):
    def __init__(self, net_structure, input_data, optimizer_method, learning_rate, batch_size):
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
            print ('%s neurons in layer_' % (self.neurons['layer' + str(i + 1)]) + str(i + 1) + '\n')


# SESSION

# 1) Setting the data parameters. Building up the neural network.
data = Data(input_file, batch_size)
net = TFNet(net_structure, data, optimizer_method, learning_rate, batch_size)

# 2) Running the session.
with tf.Session() as sess:
    # Creating the file with cost statistics, writing initial parameters.
    cost_stats = open(net.cost_stats_file, 'w+')
    cost_stats.write('# net')
    for i in net_structure:
        cost_stats.write('_%s' % (net_structure[i][0]))
    cost_stats.write('\n# learning rate = %s\n' % learning_rate + \
                     '# max iteration limit = %s\n' % max_epochs + \
                     '# batch size = %s\n' % batch_size)
    cost_stats.write('iteration,cost\n')

    # Initializing variables.
    sess.run(net.init)

    # Initiating the session run for the specified number of iterations.
    for e in range(max_epochs):
        for batch, batch_brightness in data.batches:
            sess.run(net.optimizer, feed_dict={data.nn_genotypes: batch, data.nn_brightness: batch_brightness})

        # Write down the outputs every 10th iteration.
        if e % 10 == 0:

            # Extracting net cost function output.
            to_plot_predicted = np.zeros(data.batch_number * batch_size)
            figure_name = output_folder + 'figures/net'
            for i in net_structure:
                figure_name += '_%s' % (net_structure[i][0])
            figure_name += '_iteration_%05d' % e

            costs = 0
            for index, (batch, batch_brightness) in enumerate(data.batches):
                cost_value, l3_value = sess.run([net.cost, net.output['layer3']],
                                                feed_dict={data.nn_genotypes: batch,
                                                           data.nn_brightness: batch_brightness})
                costs += cost_value

                to_plot_predicted[(index * batch_size):((index + 1) * batch_size)] = l3_value.reshape(batch_size)

            costs /= data.batch_number

            # Plotting observed versus predicted brightness. Saving the plot locally to a temp_fig  _file.
            print('Iteration %s: cost=%.7f' % (e, costs))
            cost_stats.write('%s,%s\n' % (e, costs))
            fig = plt.figure(figsize=(6, 6))
            ax = plt.subplot(111)
            density_plot(data.to_plot_observed, to_plot_predicted)
            format_plot(ax, e, costs)
            plt.savefig(temp_fig_file)
            plt.close('all')

            # Saving ckpt file and sending it and figure file to s3://landscapes-tensorflow.
            net.saver.save(sess, temp_model_ckpt_file)
            with open(temp_model_ckpt_file, 'rb') as ckpt:
                s3.Bucket('landscapes-tensorflow').put_object(Key=figure_name + '.ckpt', Body=ckpt)
            with open(temp_fig_file, 'rb') as figure_file:
                s3.Bucket('landscapes-tensorflow').put_object(Key=figure_name + '.png', Body=figure_file)

            # Saving layer1 inputs and layer3 outputs and sending those to s3://landscapes-tensorflow.
            layer1_inputs = np.zeros(data.batch_number * batch_size)
            layer3_outputs = np.zeros(data.batch_number * batch_size)

            for index, (batch, batch_brightness) in enumerate(data.batches):
                l1_values, l3_values = sess.run([net.input['layer1'], net.input['layer3']],
                                                feed_dict={data.nn_genotypes: batch,
                                                           data.nn_brightness: batch_brightness})
                layer1_inputs[(index * batch_size):((index + 1) * batch_size)] = l1_values.reshape(batch_size)
                layer3_outputs[(index * batch_size):((index + 1) * batch_size)] = l3_values.reshape(batch_size)

            neuronal_values = pd.DataFrame()
            neuronal_values['layer1_inputs'] = layer1_inputs
            neuronal_values['layer3_outputs'] = layer3_outputs

            neuronal_values_filename = output_folder + 'neuronal_values_iteration_%s.csv' % e
            neuronal_values.to_csv(neuronal_values_filename, index=False)
            with open(neuronal_values_filename, 'rb') as f:
                s3.Bucket('landscapes-tensorflow').put_object(Key=neuronal_values_filename, Body=f)

        # Reshuffling the data with the specified reshuffling frequency.
        if e % reshuffling_frequency == 0:

            # Saving parameters before reshuffling
            cost_stats.close()
            with open(cost_stats_file, 'rb') as cost_stats:
                s3.Bucket('landscapes-tensorflow').put_object(Key=cost_stats_file, Body=cost_stats)
            cost_stats = open(cost_stats_file, 'a')

            weights = sess.run(
                [net.weights['layer1'], net.biases['layer1'], net.weights['layer2'], net.biases['layer2'],
                 net.weights['layer3'], net.biases['layer3']],
                feed_dict={data.nn_genotypes: batch,
                           data.nn_brightness: batch_brightness})

            mutations_weights = pd.DataFrame()
            mutations_weights["mutation"] = data.unique_mutations
            mutations_weights["weight"] = weights[0].reshape(len(data.unique_mutations))
            mutations_weights_filename = output_folder + 'unique_mutations_scores_iteration_%s.csv' % e
            mutations_weights.to_csv(mutations_weights_filename, index=False)
            with open(mutations_weights_filename, 'rb') as mutations_weights:
                s3.Bucket('landscapes-tensorflow').put_object(Key=mutations_weights_filename,
                                                              Body=mutations_weights)

            if e != 0:
                # reshuffling data
                data.reshuffle()

    cost_stats.close()

    # sending cost_stats file to s3://landscapes-tensorflow
    with open(cost_stats_file, 'rb') as cost_stats:
        s3.Bucket('landscapes-tensorflow').put_object(Key=cost_stats_file, Body=cost_stats)
