'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import torch
import torch.nn as nn
import os, sys
import numpy as np
from time import time
from utils import metrics
from torch.optim import lr_scheduler
from utils.model_utils import save_loss, CheckpointSaver
from data.dataloader import load_dataset


def train(framework, loaders, opts):
    if opts.checkpoint_file == "none":
        resume_run = False
        experiment_dir = os.path.join(opts.save_dir, opts.experiment_name)
        train_dir = os.path.join(experiment_dir, f'training')

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if os.listdir(train_dir):
            raise IOError('train_dir is expected to be empty for new experiments. '
                          f'{train_dir} is not empty! Aborting...')

        # determine backup folder path and create it if necessary
        backup_dir = os.path.join(
            opts.backup_dir,
            opts.experiment_name + "/training"
        )
        # create the backup dir for storing validation best snapshots
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        # save command-line arguments to train and backup dir
        log_file = os.path.join(train_dir, "log_file.log")
        my_log = open(log_file, 'w')
        my_log.write("Starting training")
        my_log.close()


    else:
        resume_run = True
        train_dir = os.path.split(opts.checkpoint_file)[0]

        # load checkpoint and corresponding args.json overwriting command-line args
        checkpoint_file = opts.checkpoint_file  # backup checkpoint file path
        backup_dir = os.path.join(opts.backup_dir,
                                  opts.experiment_name + "/training")
        log_file = os.path.join(train_dir, "log_file.log")

    if len(opts.gpu_ids) <= 1:
        device = torch.device('cuda:{}'.format(opts.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')
        framework.model.to(device)

    if len(opts.gpu_ids) > 1:
        device_ids = list(opts.gpu_ids)
        framework.model = nn.DataParallel(framework.model, device_ids=device_ids)

    tic = time()
    val_history = {'loss': [], 'mean_IoU': []}
    train_history = {'loss': []}

    scheduler = lr_scheduler.ReduceLROnPlateau(framework.optimizer, 'min' if opts.num_classes > 1 else 'max', patience=opts.scheduler_patience)

    stats = {
        'train_losses': [], 'train_losses_epochs': [],
        'val_losses': [], 'val_ious': [], 'val_ious_epochs': [],
        'best_val_iou': 0., 'best_val_epoch': 0.,
        'resume_epochs': []
    }

    if resume_run:
        if opts.verbose:
            print('resuming training...')

        # load epoch, step, state_dict of model, optimizer as well as best val
        # acc and step
        checkpoint = torch.load(checkpoint_file)
        resume_epoch = checkpoint['epoch']
        framework.model.load_state_dict(checkpoint['model'])
        framework.optimizer.load_state_dict(checkpoint['optimizer'])

        stats['train_losses'] = checkpoint['train_losses']
        stats['train_losses_epochs'] = checkpoint['train_losses_epochs']
        stats['val_losses'] = checkpoint['val_losses']
        stats['val_ious'] = checkpoint['val_ious']
        stats['val_ious_epochs'] = checkpoint['val_ious_epochs']
        stats['best_val_iou'] = checkpoint['best_val_iou']
        stats['best_val_epoch'] = checkpoint['best_val_epoch']
        stats['resume_epochs'] = checkpoint['resume_epochs']
        stats['resume_epochs'].append(resume_epoch)
    else:
        if opts.verbose:
            print('starting training!')
        resume_epoch = 0

    saver = CheckpointSaver(save_dir=train_dir,
                            backup_dir=backup_dir)

    for i in range(resume_epoch, opts.n_epochs):
        my_log = open(log_file, 'a+')
        my_log.write("\n***************")
        if opts.verbose:
            print("**************")
        for phase in ['train', 'val']:
            epoch_loss = 0.0
            val_meanIoU = 0.0
            if phase == 'train':
                framework.model.train()
            else:
                framework.model.eval()
            for n_iter, (X, y) in enumerate(load_dataset(opts)[phase]):
                if torch.cuda.is_available():
                    X = X.cuda().float()

                    y = y.cuda()
                framework.optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = framework.model.forward(X)
                    loss = framework.loss(y_pred, torch.squeeze(y,1).long())
                    if phase == 'train':
                        loss.backward()
                        framework.optimizer.step()
                epoch_loss += loss.item()
                if phase == 'val':
                    y = y.cpu().numpy()
                    y_hat = y_pred.cpu().numpy()
                    y_hat = np.argmax(y_hat, axis=1)
                    batch_meanIoU = 0
                    num_samples, _, _ = y.shape
                    for j in range(num_samples):
                        batch_meanIoU += metrics.mean_IoU(y_hat[j], y[j])
                    batch_meanIoU /= num_samples
                    val_meanIoU += batch_meanIoU

            # save if notice improvement
            loss_str = "\n{} loss: {} Epoch number: {} time: {}".format(
                str(phase),
                str(epoch_loss),
                str(i),
                str(int(time() - tic))
            )
            my_log = open(log_file, 'a+')
            my_log.write(loss_str)
            if opts.verbose:
                print(loss_str)

            if phase == 'train':
                stats['train_losses'].append(epoch_loss)
                stats['train_losses_epochs'].append(i)
                train_history['loss'].append(epoch_loss)
            else:
                if n_iter > 0:
                    val_meanIoU /= (n_iter + 1)
                val_history['loss'].append(epoch_loss)
                val_history['mean_IoU'].append(val_meanIoU)
                stats['val_losses'].append(epoch_loss)
                stats['val_ious_epochs'].append(i)

                scheduler.step(epoch_loss)
                text = "\nVal meanIoU: {}".format(str(val_meanIoU))
                my_log = open(log_file, 'a+')
                my_log.write(text)
                if opts.verbose:
                    print(text)
                is_best = False
                if val_meanIoU > stats['best_val_iou']:
                    is_best = True
                    stats['best_val_roc'] = val_meanIoU
                    stats['best_val_epoch'] = i

                checkpoint = {
                    'params': opts,
                    'epoch': i,
                    'model': framework.model.state_dict(),
                    'optimizer': framework.optimizer.state_dict(),

                }
                for k, v in stats.items():
                    checkpoint[k] = v
                saver.save(state=checkpoint, is_best=is_best,
                           checkpoint_name='checkpoint')
        sys.stdout.flush()
    if opts.verbose:
        print("Training was successfully finished.")
    my_log = open(log_file, 'a+')
    my_log.write("\nTraining was successfully finished.")
    save_loss(train_history['loss'], val_history['loss'], backup_dir, "loss_figure")
    save_loss(train_history['loss'], val_history['loss'], train_dir, "loss_figure")
    return framework, train_history, val_history