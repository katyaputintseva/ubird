import tensorflow as tf
import numpy as np
from classes import Data, TFNet
from matplotlib import pyplot as plt
import pandas as pd
from functions import density_plot
from functions import format_plot

output_folder = 'output/'
cost_stats_file = output_folder + 'cost_stats.txt'
temp_model_ckpt_file = output_folder + 'model.ckpt'
input_file = '../data/amino_acid_genotypes_to_brightness_short.tsv'
batch_size = 10
net_structure = {'layer1': (1, 'tf.tanh'),
                 'layer2': (3, 'tf.tanh'),
                 'layer3': (1, 'tf.tanh')}

optimizer_method = 'tf.train.AdagradOptimizer'
learning_rate = 0.1
max_epochs = 1000
reshuffling_frequency = 10

data = Data(input_file, batch_size)
net = TFNet(net_structure, data, optimizer_method, learning_rate, batch_size)

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
            plt.savefig(figure_name)
            plt.close('all')

            # Saving ckpt file and sending it and figure file to s3://landscapes-tensorflow.
            net.saver.save(sess, temp_model_ckpt_file)

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

        # Reshuffling the data with the specified reshuffling frequency.
        if e % reshuffling_frequency == 0:

            # Saving parameters before reshuffling
            cost_stats.close()
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

            if e != 0:
                # reshuffling data
                data.reshuffle()

    cost_stats.close()
