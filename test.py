import matplotlib

matplotlib.use('Agg')
from classes import *
from variables import *
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

data = Data(input_file, batch_size)
net = TFNet(net_structure, data, optimizer_method, learning_rate, batch_size, cost_stats_file)

saver = tf.train.Saver()

with tf.Session() as sess:
    # Initializing variables.
    sess.run(net.init)

    # Initiating the session run for the specified number of iterations.

    saver.restore(sess, "~/Desktop/test/figures/net_1_1_3_iteration_06360.ckpt")

    for batch, batch_brightness in data.batches:
        sess.run(net.optimizer, feed_dict={data.nn_genotypes: batch, data.nn_brightness: batch_brightness})

        # Extracting net cost function output.
        to_plot_predicted = np.zeros(data.batch_number * batch_size)
        figure_name = output_folder + 'figures/net'
        for i in net_structure:
            figure_name += '_%s' % (net_structure[i][0])

        costs = 0
        for index, (batch, batch_brightness) in enumerate(data.batches):
            cost_value, l3_value = sess.run([net.cost, net.output['layer3']],
                                            feed_dict={data.nn_genotypes: batch,
                                                       data.nn_brightness: batch_brightness})
            costs += cost_value

            to_plot_predicted[(index * batch_size):((index + 1) * batch_size)] = l3_value.reshape(batch_size)

        costs /= data.batch_number

        # Plotting observed versus predicted brightness. Saving the plot locally to a temp_fig  _file.
        print('Cost=%.7f' % (costs))
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111)
        density_plot(data.to_plot_observed, to_plot_predicted)
        format_plot(ax, 1, costs)
        plt.savefig(figure_name)
        plt.close('all')
