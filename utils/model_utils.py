'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import os
import torch 
import shutil
import matplotlib.pyplot as plt


class CheckpointSaver(object):
    def __init__(self, save_dir, backup_dir):
        self.save_dir = save_dir
        self.backup_dir = backup_dir

    def save(self, state, is_best, checkpoint_name='checkpoint'):
        checkpoint_path = os.path.join(self.save_dir,
                                       '{}.pth.tar'.format(checkpoint_name))
        try:
            shutil.copyfile(
                checkpoint_path,
                '{}_bak'.format(checkpoint_path)
            )
        except IOError:
            pass
        torch.save(state, checkpoint_path)
        if is_best:
            try:
                shutil.copyfile(
                    os.path.join(self.backup_dir,
                                '{}_best.pth.tar'.format(checkpoint_name)),
                    os.path.join(self.backup_dir,
                                '{}_best.pth.tar_bak'.format(checkpoint_name))
                )
            except IOError:
                pass
            shutil.copyfile(
                checkpoint_path,
                os.path.join(self.backup_dir,
                             '{}_best.pth.tar'.format(checkpoint_name))
            )


def save_loss(train_loss, val_loss, save_dir, name ='loss_plots'):
    """
    :param train_loss: train losses in different epochs
    :param val_loss: validation losses in different epochs
    :return:
    """
    plt.yscale('log')
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper right')
    plt.savefig(save_dir + name + '.png')