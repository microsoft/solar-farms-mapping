'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

plt.rcParams.update({'figure.max_open_warning': 0})


def plot_loss(train_loss, val_loss):
    """
    :param train_loss: train losses in different epochs
    :param val_loss: validation losses in different epochs
    :return:
    """
    plt.yscale('log')
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper right')
    plt.show()


def plot_sample(image, pred, gt, file_name):
    '''
    Plots and a slice with all available annotations
    '''
    flatui = ["#3498db", "#FFD700"]
    color_map = ListedColormap(sns.color_palette(flatui).as_hex())

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor=color_map(0), markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='Solar Installation', markerfacecolor=color_map(1), markersize=15)]

    fig = plt.figure(figsize=(18, 18))

    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('RGB of Input')

    plt.subplot(1, 4, 2)
    plt.imshow(gt, cmap=color_map, vmin=0, vmax=1)
    plt.title('Ground Truth')

    plt.subplot(1, 4, 3)
    plt.imshow(pred, cmap=color_map, vmin=0, vmax=1)
    plt.title('Solar Pred')

    plt.subplot(1, 4, 4)
    plt.imshow(image)
    plt.imshow(pred, alpha=0.5,  interpolation='none', cmap=color_map, vmin=0, vmax=1)
    plt.title('Prediction Overlay')

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    fig.savefig(file_name, bbox_inches='tight')