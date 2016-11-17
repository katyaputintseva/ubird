import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

# Functions for plotting. The first is to actually plot, the second is to make the plot readable.
def density_plot(x, y):
    ''' x = observed, y = predicted '''
    x = x[(~np.isnan(x)) & (~np.isnan(y))]
    y = y[(~np.isnan(x)) & (~np.isnan(y))]

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


# Function for shaping the data. It performs reshuffling of the data,
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
