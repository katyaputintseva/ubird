import matplotlib
matplotlib.use('Agg')
from classes import *
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
