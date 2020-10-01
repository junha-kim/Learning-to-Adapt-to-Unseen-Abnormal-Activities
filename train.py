import time
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import wandb
from evaluation import *
from utils import *


def train_iter(model, inputs, labels, criterion, optimizer):
    output = model(inputs, model.parameters())
    loss = criterion(output, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def adam(args, beta1=0.9, beta2=0.999, eps=1e-8):  # grad, params, m_prev, v_prev, t):
    grad, params, m_prev, v_prev, inner_lr, t = args[0], args[1], args[2], args[3], args[4], args[5]
    m = beta1 * m_prev + (1 - beta1) * grad
    v = beta2 * v_prev + (1 - beta2) * (grad ** 2)
    m_ = m / (1 - beta1 ** t)
    v_ = v / (1 - beta2 ** t)

    return params - inner_lr * m_ / (eps + torch.sqrt(v_)), m, v


def pretrain(model, trainloader, validloader, optimizer, criterion,
             args, device=torch.device('cuda'), model_dir='pretrain',
             total_iters=500, valid_step=10, scheduler=None, save_chpt=True):

    all_score, all_gt, all_frame, _ = test_with_dataloader(model, validloader)
    auc = calc_auc_per_video(all_score, all_gt, all_frame)

    best_auc = auc
    best_iters = -1
    print('Best So Far :    auc={} at iter {}'.format(best_auc, best_iters))

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    model_path = None
    total_loss = 0.0
    iters = 0
    while iters < total_iters:
        for data in trainloader:
            iters += 1
            inputs, labels = data
            inputs = inputs.view(-1, inputs.size(-1)).to(device)
            labels = labels.view(-1).int().to(device)
            total_loss += train_iter(model, inputs, labels, criterion, optimizer)

            if iters % valid_step == 0:
                all_score, all_gt, all_frame, _ = test_with_dataloader(model, validloader)
                auc = calc_auc_per_video(all_score, all_gt, all_frame)
                if scheduler is not None:
                    scheduler.step(auc)

                if best_auc < auc:
                    if save_chpt:
                        if model_path is not None:
                            os.remove(model_path)
                        file_name = '{}iters_exp{}_seed{}_lr{}_split{}.pkl'.format(iters, args.exp, args.seed,
                                                                                   args.inner_lr, args.split_num)
                        model_path = os.path.join(model_dir, file_name)
                        torch.save(model.state_dict(), model_path)

                    best_auc = auc
                    best_iters = iters
                    print('Best So Far :    auc={} at iter {}'.format(best_auc, best_iters))
    if save_chpt:
        print("{} is saved.".format(file_name))
    return best_auc


def metatrain(meta_trainer, args, model_path='meta_model', lr_path = 'lr', save_step=100):
    outer_epoch = args.outer_epoch
    save_chpt = args.save_chpt
    torch.set_printoptions(precision=20)
    meta_trainer.net.train()
    start_time = time.time()
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    if not os.path.isdir(lr_path):
        os.mkdir(lr_path)
    if save_chpt:
        print('Saved checkpoint file name will be {}'+'epochs_exp{}_seed{}_lr{}_split{}.pkl'.format(
                    args.exp, args.seed, args.inner_lr, args.split_num))
    for epoch in range(outer_epoch):
        test_loss = meta_trainer.outer_loop()
        if args.wandb:
            wandb.log({"loss": test_loss})
        if epoch % save_step == 0:
            print('Epoch {}, test_loss: {}, time per print: {}'.format(epoch, test_loss, time.time()-start_time))
            start_time = time.time()
            if save_chpt:
                file_name = '{}epochs_exp{}_seed{}_lr{}_split{}.pkl'.format(
                    epoch, args.exp, args.seed, args.inner_lr, args.split_num)
                torch.save(meta_trainer.net.state_dict(), os.path.join(model_path, file_name))
                if args.meta_sgd:
                    with open(os.path.join(lr_path, file_name), 'wb') as f:
                        pickle.dump(meta_trainer.lr_params, f)






