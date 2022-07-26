import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from training.common import AverageMeter, one_hot, cut_input, get_embed, data_aug, sym_kld
from tqdm import tqdm
from sklearn import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = 1e-8

def standard_normalization(features):
    mean = features.mean(dim=-1, keepdim=True).detach()
    std = features.std(dim=-1, keepdim=True).detach()
    return (features - mean) / (std + eps)

def valuation(args, train_loader, idx_map, model, estimator, pred_train_model_val, y_train, measurement, labels_mix, log_dir, epoch, logger=None):
    logger.log('Valuating with trained valuation model...')

    model.eval()
    estimator.eval()

    y_train_onehot = np.eye(len(np.unique(y_train.numpy())))[y_train.numpy().astype(int)]
    y_pred_diff = np.abs(y_train_onehot - pred_train_model_val.numpy())

    clean_idx = (labels_mix == 1)
    noisy_idx = (labels_mix == 0)

    all_est_value = []

    for i, (tokens, labels, indices) in enumerate(tqdm(train_loader)):
        # Pre-processing
        tokens, labels = tokens.to(device), labels.to(device)
        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)

        if args.no_diff:
            y_pred_diff_batch = torch.zeros_like(torch.Tensor(y_pred_diff[idx_map[indices].numpy()])).to(device)
        else:
            y_pred_diff_batch = torch.Tensor(y_pred_diff[idx_map[indices].numpy()]).to(device)
        y_train_onehot_batch = torch.Tensor(y_train_onehot[idx_map[indices].numpy()]).to(device)
        measurement_batch = torch.Tensor(measurement[idx_map[indices].numpy()]).to(device)

        # Data valuation
        with torch.no_grad():
            logit, penul = model(tokens, get_penul=True)

        if args.measurement:
            input_estim = torch.cat([measurement_batch, y_train_onehot_batch], dim=-1)
        elif args.logit:
            if args.pre:
                logit_orig = pred_train_model_val[idx_map[indices]].to(device)
                input_estim = torch.cat([logit_orig, y_train_onehot_batch], dim=-1)  # (B, dim + class)
            else:
                input_estim = torch.cat([logit, y_train_onehot_batch], dim=-1)  # (B, dim + class)
        elif args.logit2:
            input_estim_rev = logit[torch.arange(len(logit)), labels].unsqueeze(-1)
            input_estim = input_estim_rev.repeat(1, 6)
        else:
            input_estim = torch.cat([penul, y_train_onehot_batch], dim=-1)  # (B, dim + class)

        if args.normalize:
            input_estim = standard_normalization(input_estim)
            y_pred_diff_batch = standard_normalization(y_pred_diff_batch)

        ##### Inference of value estimator
        with torch.no_grad():
            if args.measurement2:
                est_dv_curr = estimator(input_estim, measurement_batch)
            else:
                est_dv_curr = estimator(input_estim, y_pred_diff_batch)

        all_est_value.append(est_dv_curr.cpu())

    all_est_value = torch.cat(all_est_value, dim=0)
    clean_est_value = all_est_value[clean_idx]
    noisy_est_value = all_est_value[noisy_idx]

    msg = '[est_dv %.3f] [clean_dv %.3f] [noisy_dv %.3f]' \
          % (all_est_value.mean().item(), clean_est_value.mean().item(), noisy_est_value.mean().item())

    save_dir = './' + log_dir + '/all_est_value_epoch' + str(epoch) + '.npy'
    np.save(save_dir, all_est_value.numpy())

    if logger:
        logger.log(msg)
    else:
        print(msg)

def valuation_all(args, penul_train_model_train, model, estimator, pred_train_model_val, y_train, measurement, labels_mix, log_dir, epoch, logger=None):
    logger.log('Valuating with trained valuation model...')

    model.eval()
    estimator.eval()

    y_train_onehot = np.eye(len(np.unique(y_train.numpy())))[y_train.numpy().astype(int)]
    y_pred_diff = np.abs(y_train_onehot - pred_train_model_val.numpy())

    clean_idx = (labels_mix == 1)
    noisy_idx = (labels_mix == 0)

    # Pre-processing
    labels = y_train.to(device)
    if args.no_diff:
        y_pred_diff_batch = torch.zeros_like(torch.Tensor(y_pred_diff)).to(device)
    else:
        y_pred_diff_batch = torch.Tensor(y_pred_diff).to(device)
    y_train_onehot_batch = torch.Tensor(y_train_onehot).to(device)
    measurement_batch = torch.Tensor(measurement).to(device)

    # Data valuation
    logit = model.net_cls(penul_train_model_train.to(device))

    # Data valuation
    if args.measurement:
        input_estim = torch.cat([measurement_batch, y_train_onehot_batch], dim=-1)
    elif args.logit:
        if args.pre:
            logit_orig = pred_train_model_val.to(device)
            input_estim = torch.cat([logit_orig, y_train_onehot_batch], dim=-1)  # (B, dim + class)
        else:
            input_estim = torch.cat([logit, y_train_onehot_batch], dim=-1)  # (B, dim + class)
    elif args.logit2:
        input_estim_rev = logit[torch.arange(len(logit)), labels].unsqueeze(-1)
        input_estim = input_estim_rev.repeat(1, 6)
    else:
        input_estim = torch.cat([penul, y_train_onehot_batch], dim=-1)  # (B, dim + class)

    if args.normalize:
        input_estim = standard_normalization(input_estim)
        y_pred_diff_batch = standard_normalization(y_pred_diff_batch)

    ##### Inference of value estimator
    with torch.no_grad():
        if args.measurement2:
            est_dv_curr = estimator(input_estim, measurement_batch)
        else:
            est_dv_curr = estimator(input_estim, y_pred_diff_batch)

    all_est_value = est_dv_curr.cpu()
    clean_est_value = all_est_value[clean_idx]
    noisy_est_value = all_est_value[noisy_idx]

    msg = '[est_dv %.3f] [clean_dv %.3f] [noisy_dv %.3f]' \
          % (all_est_value.mean().item(), clean_est_value.mean().item(), noisy_est_value.mean().item())

    save_dir = './' + log_dir + '/all_est_value_epoch' + str(epoch) + '.npy'
    np.save(save_dir, all_est_value.numpy())

    if logger:
        logger.log(msg)
    else:
        print(msg)

def train_dvrl_orig(args, train_loader, idx_map, val_loader, model, estimator, model_optimizer, est_optimizer, pred_train_model_train,
                    pred_train_model_val, y_train, pred_valid_model_train, y_valid, measurement, epoch=0, logger=None,
                    writer=None):
    # Encourages exploration
    threshold_u = 0.9

    # Baseline performance
    train_iter = iter(train_loader)

    model.train()
    estimator.train()

    losses = dict()
    losses['est_dv_curr'] = AverageMeter()
    losses['dv_max'] = AverageMeter()
    losses['dv_min'] = AverageMeter()

    losses['dve_loss'] = AverageMeter()
    losses['reg'] = AverageMeter()
    losses['reward_curr'] = AverageMeter()

    n_iter = (epoch - 1) * args.outer_iterations
    y_train_onehot = np.eye(len(np.unique(y_train.numpy())))[y_train.numpy().astype(int)]
    y_pred_diff = np.abs(y_train_onehot - pred_train_model_val.numpy())
    reward_final, valid_move = 0.0, 0.0

    for batch_idx in tqdm(range(args.outer_iterations)):
        try:
            tokens, labels, indices = train_iter.next()
        except:
            train_iter = iter(train_loader)
            tokens, labels, indices = train_iter.next()

        # Pre-processing
        tokens, labels = tokens.to(device), labels.to(device)
        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)

        if args.no_diff:
            y_pred_diff_batch = torch.zeros_like(torch.Tensor(y_pred_diff[idx_map[indices].numpy()])).to(device)
        else:
            y_pred_diff_batch = torch.Tensor(y_pred_diff[idx_map[indices].numpy()]).to(device).abs()
        y_train_onehot_batch = torch.Tensor(y_train_onehot[idx_map[indices].numpy()]).to(device)
        measurement_batch = torch.Tensor(measurement[idx_map[indices].numpy()]).to(device)

        # Data valuation
        with torch.no_grad():
            logit, penul = model(tokens, get_penul=True)

        if args.measurement:
            input_estim = torch.cat([measurement_batch, y_train_onehot_batch], dim=-1)
        elif args.logit:
            if args.pre:
                logit_orig = pred_train_model_train[idx_map[indices]].to(device)
                input_estim = torch.cat([logit_orig, y_train_onehot_batch], dim=-1)  # (B, dim + class)
            else:
                input_estim = torch.cat([logit, y_train_onehot_batch], dim=-1)  # (B, dim + class)
        elif args.logit2:
            input_estim_rev = logit[torch.arange(len(logit)), labels].unsqueeze(-1)
            input_estim = input_estim_rev.repeat(1, 6)
        else:
            input_estim = torch.cat([penul, y_train_onehot_batch], dim=-1)  # (B, dim + class)

        if args.normalize:
            input_estim = standard_normalization(input_estim)
            y_pred_diff_batch = standard_normalization(y_pred_diff_batch)
            measurement_batch_n = standard_normalization(measurement_batch.clone())

        ##### Inference of value estimator
        if args.measurement2:
            est_dv_curr = estimator(input_estim, measurement_batch_n)
        else:
            est_dv_curr = estimator(input_estim, y_pred_diff_batch)

        ##### Sample the mask
        if args.continuous:
            sel_prob_curr = est_dv_curr
        else:
            sel_prob_curr = np.random.binomial(1, est_dv_curr.data.cpu().numpy(), est_dv_curr.data.cpu().shape)

        # Update model with obtained sample weights
        train(args, [tokens, labels, sel_prob_curr], model, model_optimizer)

        # Reward computation
        pred_valid, loss_valid, indices_partial = predicts_partial(args, val_loader, model)
        y_valid_partial = y_valid[indices_partial].cpu().numpy()

        y_valid_hat = pred_valid.softmax(dim=-1).numpy()
        y_valid_hat_base = pred_valid_model_train[indices_partial].softmax(dim=-1).numpy()

        if args.perf_metric == 'auc':
            dvrl_perf = metrics.roc_auc_score(y_valid_partial, y_valid_hat, multi_class='ovr')
            valid_perf = metrics.roc_auc_score(y_valid_partial, y_valid_hat_base, multi_class='ovr')
        elif args.perf_metric == 'accuracy':
            dvrl_perf = metrics.accuracy_score(y_valid_partial, np.argmax(y_valid_hat, axis=1))
            valid_perf = metrics.accuracy_score(y_valid_partial, np.argmax(y_valid_hat_base, axis=1))
        else:
            dvrl_perf = -metrics.log_loss(y_valid_partial, y_valid_hat)
            valid_perf = -metrics.log_loss(y_valid_partial, y_valid_hat_base)

        if args.base:
            reward_curr = dvrl_perf - valid_perf
        elif args.move_base:
            reward_curr = dvrl_perf - valid_move

            if batch_idx < args.window_size:
                window_size = (batch_idx + 1)
            else:
                window_size = args.window_size

            valid_move *= ((window_size - 1) / window_size)
            valid_move += (dvrl_perf / window_size)
        else:
            reward_curr = dvrl_perf

        if args.window_size > 0:
            if batch_idx < args.window_size:
                window_size = (batch_idx + 1)
            else:
                window_size = args.window_size

            reward_final *= ((window_size - 1) / window_size)
            reward_final += (reward_curr / window_size)
        else:
            reward_final = reward_curr

        # Generator loss (REINFORCE algorithm)
        sel_prob_curr = torch.Tensor(sel_prob_curr).to(device)
        prob = torch.sum(sel_prob_curr * torch.log(est_dv_curr + eps) + (1 - sel_prob_curr) * torch.log(1 - est_dv_curr + eps)) # it would be minus

        dve_loss = -1 * reward_final * prob
        if args.no_regular:
            reg = torch.Tensor([0]).to(device)
        else:
            reg = args.lambda_reg * (torch.clamp(torch.mean(est_dv_curr) - threshold_u, min=0) + torch.clamp(
                (1 - threshold_u) - torch.mean(est_dv_curr), min=0))
        # Update Estimator
        est_optimizer.zero_grad()
        (dve_loss + reg).backward()
        est_optimizer.step()

        # Logging
        writer.add_scalars('rl optimization', {'reward_curr': reward_curr,
                                               'dve_loss': dve_loss.mean().detach().cpu().item(),
                                               'reg': reg.mean().detach().cpu().item(),
                                               'prob': prob.mean().detach().cpu().item(),
                                               'dvrl_perf': dvrl_perf,
                                               'base_perf': valid_perf}, n_iter)
        writer.add_scalars('data values', {'est_dv_curr': est_dv_curr.mean().detach().cpu().item(),
                                               'est_dv_max': est_dv_curr.max().detach().cpu().item(),
                                               'est_dv_min': est_dv_curr.min().detach().cpu().item()}, n_iter)

        n_iter += 1
        losses['est_dv_curr'].update(est_dv_curr.mean().detach().cpu().item(), len(y_valid_hat))
        losses['dv_max'].update(est_dv_curr.max().detach().cpu().item(), len(y_valid_hat))
        losses['dv_min'].update(est_dv_curr.min().detach().cpu().item(), len(y_valid_hat))

        losses['reward_curr'].update(reward_curr, len(y_valid_hat))
        losses['dve_loss'].update(dve_loss.mean().detach().cpu().item(), len(y_valid_hat))
        losses['reg'].update(reg.mean().detach().cpu().item(), len(y_valid_hat))

    msg = '[est_dv %.3f] [max_dv %.3f] [min_dv %.3f] [reward %.3f] [loss_dve %.3f] [reg %.3f]' \
          % (losses['est_dv_curr'].average, losses['dv_max'].average, losses['dv_min'].average,
             losses['reward_curr'].average, losses['dve_loss'].average, losses['reg'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)


def train_dvrl_orig_epoch(args, train_loader, idx_map, val_loader, model, estimator, model_optimizer, est_optimizer, pred_train_model_train,
                    pred_train_model_val, y_train, pred_valid_model_train, y_valid, measurement, epoch=0, logger=None,
                    writer=None):
    # Encourages exploration
    threshold_u = 0.9

    model.train()
    estimator.train()

    losses = dict()
    losses['est_dv_curr'] = AverageMeter()
    losses['dv_max'] = AverageMeter()
    losses['dv_min'] = AverageMeter()

    losses['dve_loss'] = AverageMeter()
    losses['reg'] = AverageMeter()
    losses['reward_curr'] = AverageMeter()

    n_iter = (epoch - 1) * len(train_loader)
    y_train_onehot = np.eye(len(np.unique(y_train.numpy())))[y_train.numpy().astype(int)]
    y_pred_diff = np.abs(y_train_onehot - pred_train_model_val.numpy())
    reward_final = 0

    if args.dataset == 'stsb':
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    for i, (tokens, labels, indices) in enumerate(tqdm(train_loader)):
        # Pre-processing
        tokens, labels = tokens.to(device), labels.to(device)
        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)

        if args.no_diff:
            y_pred_diff_batch = torch.zeros_like(torch.Tensor(y_pred_diff[idx_map[indices].numpy()])).to(device)
        else:
            y_pred_diff_batch = torch.Tensor(y_pred_diff[idx_map[indices].numpy()]).to(device)
        y_train_onehot_batch = torch.Tensor(y_train_onehot[idx_map[indices].numpy()]).to(device)
        measurement_batch = torch.Tensor(measurement[idx_map[indices].numpy()]).to(device)

        # Data valuation
        logit, penul = model(tokens, get_penul=True)

        if args.measurement:
            input_estim = torch.cat([measurement_batch, y_train_onehot_batch], dim=-1)
        elif args.logit:
            if args.pre:
                logit_orig = pred_train_model_train[idx_map[indices]].to(device)
                input_estim = torch.cat([logit_orig, y_train_onehot_batch], dim=-1)  # (B, dim + class)
            else:
                input_estim = torch.cat([logit.clone().detach(), y_train_onehot_batch], dim=-1)  # (B, dim + class)
        elif args.logit2:
            input_estim_rev = logit[torch.arange(len(logit)), labels].unsqueeze(-1)
            input_estim = input_estim_rev.repeat(1, 6)
        else:
            input_estim = torch.cat([penul.clone.detach(), y_train_onehot_batch], dim=-1)  # (B, dim + class)

        if args.normalize:
            input_estim = standard_normalization(input_estim)
            y_pred_diff_batch = standard_normalization(y_pred_diff_batch)
            measurement_batch_n = standard_normalization(measurement_batch.clone())

        ##### Inference of value estimator
        if args.measurement2:
            est_dv_curr = estimator(input_estim, measurement_batch_n)
        else:
            est_dv_curr = estimator(input_estim, y_pred_diff_batch)

        ##### Sample the mask
        sel_prob_curr = np.random.binomial(1, est_dv_curr.data.cpu().numpy(), est_dv_curr.data.cpu().shape)
        select_mask = torch.Tensor(sel_prob_curr[:, 0]).to(device)

        # Update model with obtained sample weights
        loss = (select_mask * criterion(logit, labels)).mean()

        est_optimizer.zero_grad()
        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

        # Reward computation
        pred_valid, loss_valid, indices_partial = predicts_partial(args, val_loader, model)
        y_valid_partial = y_valid[indices_partial].cpu().numpy()

        y_valid_hat = pred_valid.softmax(dim=-1).numpy()
        y_valid_hat_base = pred_valid_model_train[indices_partial].softmax(dim=-1).numpy()

        if args.perf_metric == 'auc':
            dvrl_perf = metrics.roc_auc_score(y_valid_partial, y_valid_hat, multi_class='ovr')
            valid_perf = metrics.roc_auc_score(y_valid_partial, y_valid_hat_base, multi_class='ovr')
        elif args.perf_metric == 'accuracy':
            dvrl_perf = metrics.accuracy_score(y_valid_partial, np.argmax(y_valid_hat, axis=1))
            valid_perf = metrics.accuracy_score(y_valid_partial, np.argmax(y_valid_hat_base, axis=1))
        else:
            dvrl_perf = -metrics.log_loss(y_valid_partial, y_valid_hat)
            valid_perf = -metrics.log_loss(y_valid_partial, y_valid_hat_base)

        if args.base:
            reward_curr = dvrl_perf - valid_perf
        else:
            reward_curr = dvrl_perf

        if args.window_size > 0:
            if batch_idx < args.window_size:
                window_size = (batch_idx + 1)
            else:
                window_size = args.window_size

            reward_final *= ((window_size - 1) / window_size)
            reward_final += (reward_curr / window_size)
        else:
            reward_final = reward_curr

        # Generator loss (REINFORCE algorithm)
        sel_prob_curr = torch.Tensor(sel_prob_curr).to(device)
        prob = torch.sum(sel_prob_curr * torch.log(est_dv_curr + eps) + (1 - sel_prob_curr) * torch.log(
            1 - est_dv_curr + eps))  # it would be minus

        dve_loss = -1 * reward_final * prob
        if args.no_regular:
            reg = torch.Tensor([0]).to(device)
        else:
            reg = args.lambda_reg * (torch.clamp(torch.mean(est_dv_curr) - threshold_u, min=0) + torch.clamp(
                (1 - threshold_u) - torch.mean(est_dv_curr), min=0))
        # Update Estimator
        est_optimizer.zero_grad()
        (dve_loss + reg).backward()
        est_optimizer.step()

        # Logging
        writer.add_scalars('rl optimization', {'reward_curr': reward_curr,
                                               'dve_loss': dve_loss.mean().detach().cpu().item(),
                                               'reg': reg.mean().detach().cpu().item(),
                                               'prob': prob.mean().detach().cpu().item(),
                                               'dvrl_perf': dvrl_perf,
                                               'base_perf': valid_perf}, n_iter)
        writer.add_scalars('data values', {'est_dv_curr': est_dv_curr.mean().detach().cpu().item(),
                                           'est_dv_max': est_dv_curr.max().detach().cpu().item(),
                                           'est_dv_min': est_dv_curr.min().detach().cpu().item()}, n_iter)

        n_iter += 1
        losses['est_dv_curr'].update(est_dv_curr.mean().detach().cpu().item(), len(y_valid_hat))
        losses['dv_max'].update(est_dv_curr.max().detach().cpu().item(), len(y_valid_hat))
        losses['dv_min'].update(est_dv_curr.min().detach().cpu().item(), len(y_valid_hat))

        losses['reward_curr'].update(reward_curr, len(y_valid_hat))
        losses['dve_loss'].update(dve_loss.mean().detach().cpu().item(), len(y_valid_hat))
        losses['reg'].update(reg.mean().detach().cpu().item(), len(y_valid_hat))

    msg = '[est_dv %.3f] [max_dv %.3f] [min_dv %.3f] [reward %.3f] [loss_dve %.3f] [reg %.3f]' \
          % (losses['est_dv_curr'].average, losses['dv_max'].average, losses['dv_min'].average,
             losses['reward_curr'].average, losses['dve_loss'].average, losses['reg'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)


def train_dvrl_orig_epoch_lin(args, est_dv_value, labels_mix, model, estimator, model_optimizer, est_optimizer, pred_train_model_train,
                              penul_train_model_train, pred_train_model_val, y_train, pred_valid_model_train, penul_valid_model_train,
                              y_valid, measurement, epoch=0, logger=None, writer=None):
    # Encourages exploration
    threshold_u = 0.9

    model.train()
    estimator.train()

    losses = dict()
    losses['est_dv_curr'] = AverageMeter()
    losses['dv_max'] = AverageMeter()
    losses['dv_min'] = AverageMeter()

    losses['dve_loss'] = AverageMeter()
    losses['reg'] = AverageMeter()
    losses['reward_curr'] = AverageMeter()

    n_iter = (epoch - 1) * args.outer_iterations
    y_train_onehot = np.eye(len(np.unique(y_train.numpy())))[y_train.numpy().astype(int)]
    y_pred_diff = np.abs(y_train_onehot - pred_train_model_val.numpy())
    penul_valid_model_train = penul_valid_model_train.to(device)
    reward_final = 0

    clean_idx = (labels_mix == 1)
    noisy_idx = (labels_mix == 0)

    if args.dataset == 'stsb':
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    est_dv_value_current = est_dv_value

    for batch_idx in tqdm(range(args.outer_iterations)):
        # Pre-processing
        labels = y_train.to(device)

        if args.no_diff:
            y_pred_diff_batch = torch.zeros_like(torch.Tensor(y_pred_diff)).to(device)
        else:
            y_pred_diff_batch = torch.Tensor(y_pred_diff).to(device)
        y_train_onehot_batch = torch.Tensor(y_train_onehot).to(device)
        measurement_batch = torch.Tensor(measurement).to(device)

        # Data valuation
        logit = model.net_cls(penul_train_model_train.to(device))

        if args.measurement:
            input_estim = torch.cat([measurement_batch, y_train_onehot_batch], dim=-1)
        elif args.logit:
            if args.pre:
                logit_orig = pred_train_model_train.to(device)
                input_estim = torch.cat([logit_orig, y_train_onehot_batch], dim=-1)  # (B, dim + class)
            else:
                input_estim = torch.cat([logit.clone().detach(), y_train_onehot_batch], dim=-1)  # (B, dim + class)
        elif args.logit2:
            input_estim_rev = logit[torch.arange(len(logit)), labels].unsqueeze(-1)
            input_estim = input_estim_rev.repeat(1, 6)
        else:
            input_estim = 0
            print("Currently, it has been not implemented yet")

        if args.normalize:
            input_estim = standard_normalization(input_estim)
            y_pred_diff_batch = standard_normalization(y_pred_diff_batch)
            measurement_batch_n = standard_normalization(measurement_batch.clone())

        ##### Inference of value estimator
        if args.measurement2:
            est_dv_curr = estimator(input_estim, measurement_batch_n)
        else:
            est_dv_curr = estimator(input_estim, y_pred_diff_batch)

        sel_prob_curr = np.random.binomial(1, est_dv_curr.data.cpu().numpy(), est_dv_curr.data.cpu().shape)
        sel_prob_curr = torch.Tensor(sel_prob_curr).to(device)
        select_mask = sel_prob_curr[:, 0]

        # Update model with obtained sample weights
        loss = (select_mask * criterion(logit, labels)).mean()

        est_optimizer.zero_grad()
        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

        # Reward computation
        with torch.no_grad():
            pred_valid = model.net_cls(penul_valid_model_train)
        y_valid_partial = y_valid.cpu().numpy()

        y_valid_hat = pred_valid.softmax(dim=-1).cpu().numpy()
        y_valid_hat_base = pred_valid_model_train.softmax(dim=-1).numpy()

        if args.perf_metric == 'auc':
            dvrl_perf = metrics.roc_auc_score(y_valid_partial, y_valid_hat, multi_class='ovr')
            valid_perf = metrics.roc_auc_score(y_valid_partial, y_valid_hat_base, multi_class='ovr')
        elif args.perf_metric == 'accuracy':
            dvrl_perf = metrics.accuracy_score(y_valid_partial, np.argmax(y_valid_hat, axis=1))
            valid_perf = metrics.accuracy_score(y_valid_partial, np.argmax(y_valid_hat_base, axis=1))
        else:
            dvrl_perf = -metrics.log_loss(y_valid_partial, y_valid_hat)
            valid_perf = -metrics.log_loss(y_valid_partial, y_valid_hat_base)

        if args.base:
            reward_curr = dvrl_perf - valid_perf
        else:
            reward_curr = dvrl_perf

        if args.window_size > 0:
            if batch_idx < args.window_size:
                window_size = (batch_idx + 1)
            else:
                window_size = args.window_size

            reward_final *= ((window_size - 1) / window_size)
            reward_final += (reward_curr / window_size)
        else:
            reward_final = reward_curr

        # Generator loss (REINFORCE algorithm)
        prob = torch.sum(sel_prob_curr * torch.log(est_dv_curr + eps) + (1 - sel_prob_curr) * torch.log(
            1 - est_dv_curr + eps))  # it would be minus

        dve_loss = -1 * reward_final * prob
        if args.no_regular:
            reg = torch.Tensor([0]).to(device)
        else:
            reg = args.lambda_reg * (torch.clamp(torch.mean(est_dv_curr) - threshold_u, min=0) + torch.clamp(
                (1 - threshold_u) - torch.mean(est_dv_curr), min=0))
        # Update Estimator
        est_optimizer.zero_grad()
        (dve_loss + reg).backward()
        est_optimizer.step()

        # Logging
        est_dv_value_current = 0.999 * est_dv_value_current + (1 - 0.999) * est_dv_curr[:, 0]

        writer.add_scalars('rl optimization', {'reward_curr': reward_curr,
                                               'dve_loss': dve_loss.mean().detach().cpu().item(),
                                               'reg': reg.mean().detach().cpu().item(),
                                               'prob': prob.mean().detach().cpu().item(),
                                               'dvrl_perf': dvrl_perf,
                                               'base_perf': valid_perf}, n_iter)
        writer.add_scalars('data values', {'est_dv_curr': est_dv_curr.mean().detach().cpu().item(),
                                           'est_dv_max': est_dv_curr.max().detach().cpu().item(),
                                           'est_dv_min': est_dv_curr.min().detach().cpu().item()}, n_iter)

        writer.add_scalars('clean values', {'est_dv_curr': est_dv_curr[clean_idx].mean().detach().cpu().item(),
                                           'est_dv_max': est_dv_curr[clean_idx].max().detach().cpu().item(),
                                           'est_dv_min': est_dv_curr[clean_idx].min().detach().cpu().item()}, n_iter)

        writer.add_scalars('noisy values', {'est_dv_curr': est_dv_curr[noisy_idx].mean().detach().cpu().item(),
                                           'est_dv_max': est_dv_curr[noisy_idx].max().detach().cpu().item(),
                                           'est_dv_min': est_dv_curr[noisy_idx].min().detach().cpu().item()}, n_iter)

        n_iter += 1
        losses['est_dv_curr'].update(est_dv_curr.mean().detach().cpu().item(), len(y_valid_hat))
        losses['dv_max'].update(est_dv_curr.max().detach().cpu().item(), len(y_valid_hat))
        losses['dv_min'].update(est_dv_curr.min().detach().cpu().item(), len(y_valid_hat))

        losses['reward_curr'].update(reward_curr, len(y_valid_hat))
        losses['dve_loss'].update(dve_loss.mean().detach().cpu().item(), len(y_valid_hat))
        losses['reg'].update(reg.mean().detach().cpu().item(), len(y_valid_hat))

    msg = '[est_dv %.3f] [max_dv %.3f] [min_dv %.3f] [reward %.3f] [loss_dve %.3f] [reg %.3f]' \
          % (losses['est_dv_curr'].average, losses['dv_max'].average, losses['dv_min'].average,
             losses['reward_curr'].average, losses['dve_loss'].average, losses['reg'].average)

    msg2 = 'MV_avg [clean_dv %.3f] [clean_max %.3f] [clean_min %.3f]' \
          % (est_dv_value_current[clean_idx].mean().item(), est_dv_value_current[clean_idx].max().item(),
             est_dv_value_current[clean_idx].min().item())

    msg3 = 'MV_avg [noisy_dv %.3f] [noisy_max %.3f] [noisy_min %.3f]' \
           % (est_dv_value_current[noisy_idx].mean().item(), est_dv_value_current[noisy_idx].max().item(),
              est_dv_value_current[noisy_idx].min().item())

    if logger:
        logger.log(msg)
        logger.log(msg2)
        logger.log(msg3)
    else:
        print(msg)
        print(msg2)
        print(msg3)

    return est_dv_value_current.data

def train_dvrl_orig_epoch_lin_backup(args, train_loader, idx_map, val_loader, model, estimator, model_optimizer, est_optimizer, pred_train_model_train,
                              penul_train_model_train, pred_train_model_val, y_train, pred_valid_model_train, penul_valid_model_train,
                              y_valid, measurement, epoch=0, logger=None, writer=None):
    # Encourages exploration
    threshold_u = 0.9

    model.train()
    estimator.train()

    losses = dict()
    losses['est_dv_curr'] = AverageMeter()
    losses['dv_max'] = AverageMeter()
    losses['dv_min'] = AverageMeter()

    losses['dve_loss'] = AverageMeter()
    losses['reg'] = AverageMeter()
    losses['reward_curr'] = AverageMeter()

    n_iter = (epoch - 1) * args.outer_iterations
    y_train_onehot = np.eye(len(np.unique(y_train.numpy())))[y_train.numpy().astype(int)]
    y_pred_diff = np.abs(y_train_onehot - pred_train_model_val.numpy())
    penul_valid_model_train = penul_valid_model_train.to(device)
    train_iter = iter(train_loader)
    reward_final = 0

    if args.dataset == 'stsb':
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    for batch_idx in tqdm(range(args.outer_iterations)):
        try:
            tokens, labels, indices = train_iter.next()
        except:
            train_iter = iter(train_loader)
            tokens, labels, indices = train_iter.next()

        # Pre-processing
        tokens, labels = tokens.to(device), y_train.to(device)

        if args.no_diff:
            y_pred_diff_batch = torch.zeros_like(torch.Tensor(y_pred_diff)).to(device)
        else:
            y_pred_diff_batch = torch.Tensor(y_pred_diff).to(device)
        y_train_onehot_batch = torch.Tensor(y_train_onehot).to(device)
        measurement_batch = torch.Tensor(measurement).to(device)

        # Data valuation
        logit = model.net_cls(penul_train_model_train.to(device))

        if args.measurement:
            input_estim = torch.cat([measurement_batch, y_train_onehot_batch], dim=-1)
        elif args.logit:
            if args.pre:
                logit_orig = pred_train_model_train.to(device)
                input_estim = torch.cat([logit_orig, y_train_onehot_batch], dim=-1)  # (B, dim + class)
            else:
                input_estim = torch.cat([logit.clone().detach(), y_train_onehot_batch], dim=-1)  # (B, dim + class)
        elif args.logit2:
            input_estim_rev = logit[torch.arange(len(logit)), labels].unsqueeze(-1)
            input_estim = input_estim_rev.repeat(1, 6)
        else:
            input_estim = 0
            print("Currently, it has been not implemented yet")

        if args.normalize:
            input_estim = standard_normalization(input_estim)
            y_pred_diff_batch = standard_normalization(y_pred_diff_batch)
            measurement_batch_n = standard_normalization(measurement_batch.clone())

        ##### Inference of value estimator
        if args.measurement2:
            est_dv_curr = estimator(input_estim, measurement_batch_n)
        else:
            est_dv_curr = estimator(input_estim, y_pred_diff_batch)

        ##### Sample the mask
        prob_save = 100000000
        sel_prob_curr_save = 0
        for i in range(1):
            sel_prob_curr = np.random.binomial(1, est_dv_curr.data.cpu().numpy(), est_dv_curr.data.cpu().shape)
            sel_prob_curr = torch.Tensor(sel_prob_curr).to(device)
            prob = torch.sum(sel_prob_curr * torch.log(est_dv_curr + eps) + (1 - sel_prob_curr) * torch.log(1 - est_dv_curr + eps))  # it would be minus
            if prob < prob_save:
                prob_save = prob
                sel_prob_curr_save = sel_prob_curr
        sel_prob_curr = sel_prob_curr_save
        select_mask = sel_prob_curr[:, 0]

        # Update model with obtained sample weights
        loss = (select_mask * criterion(logit, labels)).mean()

        est_optimizer.zero_grad()
        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

        # Reward computation
        with torch.no_grad():
            pred_valid = model.net_cls(penul_valid_model_train)
        y_valid_partial = y_valid.cpu().numpy()

        y_valid_hat = pred_valid.softmax(dim=-1).cpu().numpy()
        y_valid_hat_base = pred_valid_model_train.softmax(dim=-1).numpy()

        if args.perf_metric == 'auc':
            dvrl_perf = metrics.roc_auc_score(y_valid_partial, y_valid_hat, multi_class='ovr')
            valid_perf = metrics.roc_auc_score(y_valid_partial, y_valid_hat_base, multi_class='ovr')
        elif args.perf_metric == 'accuracy':
            dvrl_perf = metrics.accuracy_score(y_valid_partial, np.argmax(y_valid_hat, axis=1))
            valid_perf = metrics.accuracy_score(y_valid_partial, np.argmax(y_valid_hat_base, axis=1))
        else:
            dvrl_perf = -metrics.log_loss(y_valid_partial, y_valid_hat)
            valid_perf = -metrics.log_loss(y_valid_partial, y_valid_hat_base)

        if args.base:
            reward_curr = dvrl_perf - valid_perf
        else:
            reward_curr = dvrl_perf

        if args.window_size > 0:
            if batch_idx < args.window_size:
                window_size = (batch_idx + 1)
            else:
                window_size = args.window_size

            reward_final *= ((window_size - 1) / window_size)
            reward_final += (reward_curr / window_size)
        else:
            reward_final = reward_curr

        # Generator loss (REINFORCE algorithm)
        prob = torch.sum(sel_prob_curr * torch.log(est_dv_curr + eps) + (1 - sel_prob_curr) * torch.log(
            1 - est_dv_curr + eps))  # it would be minus

        dve_loss = -1 * reward_final * prob
        if args.no_regular:
            reg = torch.Tensor([0]).to(device)
        else:
            reg = args.lambda_reg * (torch.clamp(torch.mean(est_dv_curr) - threshold_u, min=0) + torch.clamp(
                (1 - threshold_u) - torch.mean(est_dv_curr), min=0))
        # Update Estimator
        est_optimizer.zero_grad()
        (dve_loss + reg).backward()
        est_optimizer.step()

        # Logging
        writer.add_scalars('rl optimization', {'reward_curr': reward_curr,
                                               'dve_loss': dve_loss.mean().detach().cpu().item(),
                                               'reg': reg.mean().detach().cpu().item(),
                                               'prob': prob.mean().detach().cpu().item(),
                                               'dvrl_perf': dvrl_perf,
                                               'base_perf': valid_perf}, n_iter)
        writer.add_scalars('data values', {'est_dv_curr': est_dv_curr.mean().detach().cpu().item(),
                                           'est_dv_max': est_dv_curr.max().detach().cpu().item(),
                                           'est_dv_min': est_dv_curr.min().detach().cpu().item()}, n_iter)

        n_iter += 1
        losses['est_dv_curr'].update(est_dv_curr.mean().detach().cpu().item(), len(y_valid_hat))
        losses['dv_max'].update(est_dv_curr.max().detach().cpu().item(), len(y_valid_hat))
        losses['dv_min'].update(est_dv_curr.min().detach().cpu().item(), len(y_valid_hat))

        losses['reward_curr'].update(reward_curr, len(y_valid_hat))
        losses['dve_loss'].update(dve_loss.mean().detach().cpu().item(), len(y_valid_hat))
        losses['reg'].update(reg.mean().detach().cpu().item(), len(y_valid_hat))

    msg = '[est_dv %.3f] [max_dv %.3f] [min_dv %.3f] [reward %.3f] [loss_dve %.3f] [reg %.3f]' \
          % (losses['est_dv_curr'].average, losses['dv_max'].average, losses['dv_min'].average,
             losses['reward_curr'].average, losses['dve_loss'].average, losses['reg'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)

def train_dvrl(args, train_loader, idx_map, val_loader, model, estimator, model_optimizer, est_optimizer,
               loss_valid_model_train, y_pred_diff, y_train_onehot, y_valid, measurement, epoch=0, logger=None, writer=None):
    # Encourages exploration
    threshold_u = 0.9
    interval = 0.1

    # Baseline performance
    train_iter = iter(train_loader)

    model.train()
    estimator.train()

    losses = dict()
    losses['est_dv_curr'] = AverageMeter()
    losses['dv_max'] = AverageMeter()
    losses['dv_min'] = AverageMeter()

    losses['dve_loss'] = AverageMeter()
    losses['reg'] = AverageMeter()
    losses['reward_curr'] = AverageMeter()

    n_iter = (epoch - 1) * args.outer_iterations

    loss_valid_model_train_new = loss_valid_model_train.clone()

    for _ in tqdm(range(args.outer_iterations)):
        try:
            tokens, labels, indices = train_iter.next()
        except:
            train_iter = iter(train_loader)
            tokens, labels, indices = train_iter.next()

        # Pre-processing
        tokens, labels = tokens.to(device), labels.to(device)
        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)

        if args.no_diff:
            y_pred_diff_batch = torch.zeros_like(torch.Tensor(y_pred_diff[idx_map[indices].numpy()])).to(device)
        else:
            y_pred_diff_batch = torch.Tensor(y_pred_diff[idx_map[indices].numpy()]).to(device)
        y_train_onehot_batch = torch.Tensor(y_train_onehot[idx_map[indices].numpy()]).to(device)
        measurement_batch = torch.Tensor(measurement[idx_map[indices].numpy()]).to(device)

        # Data valuation
        with torch.no_grad():
            logit, penul = model(tokens, get_penul=True)

        if args.measurement:
            input_estim = torch.cat([measurement_batch, y_train_onehot_batch], dim=-1)
        else:
            input_estim = torch.cat([penul, y_train_onehot_batch], dim=-1) # (B, dim + class)
            ## Debugging
            ## input_estim = torch.zeros(input_estim.size()).to(device)

        if args.normalize:
            input_estim = standard_normalization(input_estim)
            y_pred_diff_batch = standard_normalization(y_pred_diff_batch)

        ##### Inference of value estimator
        est_dv_curr = estimator(input_estim, y_pred_diff_batch)
        
        ##### Sample the mask
        sel_prob_curr = np.random.binomial(1, est_dv_curr.data.cpu().numpy(), est_dv_curr.data.cpu().shape)

        # Update model with obtained sample weights
        train(args, [tokens, labels, sel_prob_curr], model, model_optimizer)

        # Reward computation
        valid_perf = loss_valid_model_train
        pred_valid, loss_valid, indices_partial = predicts_partial(args, val_loader, model)

        valid_perf_partial = valid_perf[indices_partial]
        dvrl_perf = loss_valid

        if args.base:
            reward_curr = (dvrl_perf - valid_perf_partial).mean().to(device)
        else:
            reward_curr = (dvrl_perf).mean().to(device)

        # Update the baseline
        loss_valid_model_train_new[indices_partial] *= ((args.window_size - 1) / args.window_size)
        loss_valid_model_train_new[indices_partial] += ((1 / args.window_size) * loss_valid)

        # Generator loss (REINFORCE algorithm)
        sel_prob_curr = torch.Tensor(sel_prob_curr).to(device)
        prob = torch.sum(sel_prob_curr * torch.log(est_dv_curr + eps) + (1 - sel_prob_curr) * torch.log(1 - est_dv_curr + eps))

        dve_loss = -1 * reward_curr * prob
        if args.no_regular:
            reg = torch.Tensor([0]).to(device)
        else:
            reg = args.lambda_reg * (torch.clamp(torch.mean(est_dv_curr) - threshold_u, min=0) + torch.clamp((1 - threshold_u) - torch.mean(est_dv_curr), min=0))
            reg += args.lambda_reg * (torch.clamp(interval - torch.abs(torch.mean(est_dv_curr) - 0.5), min=0))
        # Update Estimator
        est_optimizer.zero_grad()
        (dve_loss + reg).backward()
        est_optimizer.step()

        # Logging
        writer.add_scalars('rl optimization', {'reward_curr': reward_curr.mean().detach().cpu().item(),
                                               'dve_loss': dve_loss.mean().detach().cpu().item(),
                                               'reg': reg.mean().detach().cpu().item(),
                                               'est_dv_curr': est_dv_curr.mean().detach().cpu().item(),
                                               'est_dv_max': est_dv_curr.max().detach().cpu().item(),
                                               'est_dv_min': est_dv_curr.min().detach().cpu().item(),
                                               'dvrl_perf': dvrl_perf.mean().item(),
                                               'base_perf': valid_perf_partial.mean().item()}, n_iter)

        n_iter += 1
        losses['est_dv_curr'].update(est_dv_curr.mean().detach().cpu().item(), len(dvrl_perf))
        losses['dv_max'].update(est_dv_curr.max().detach().cpu().item(), len(dvrl_perf))
        losses['dv_min'].update(est_dv_curr.min().detach().cpu().item(), len(dvrl_perf))

        losses['reward_curr'].update(reward_curr.mean().detach().cpu().item(), len(dvrl_perf))
        losses['dve_loss'].update(dve_loss.mean().detach().cpu().item(), len(dvrl_perf))
        losses['reg'].update(reg.mean().detach().cpu().item(), len(dvrl_perf))

    msg = '[est_dv %.3f] [max_dv %.3f] [min_dv %.3f] [reward %.3f] [loss_dve %.3f] [reg %.3f]' \
          % (losses['est_dv_curr'].average, losses['dv_max'].average, losses['dv_min'].average,
             losses['reward_curr'].average, losses['dve_loss'].average, losses['reg'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)

    return loss_valid_model_train_new


def train_dvrl_epoch(args, train_loader, idx_map, val_loader, model, estimator, model_optimizer, est_optimizer,
               loss_valid_model_train, y_pred_diff, y_train_onehot, y_valid, measurement, epoch=0, logger=None, writer=None):
    # Encourages exploration
    threshold_u = 0.9
    interval = 0.1

    # Baseline performance
    model.train()
    estimator.train()

    losses = dict()
    losses['est_dv_curr'] = AverageMeter()
    losses['dv_max'] = AverageMeter()
    losses['dv_min'] = AverageMeter()

    losses['dve_loss'] = AverageMeter()
    losses['reg'] = AverageMeter()
    losses['reward_curr'] = AverageMeter()

    n_iter = (epoch - 1) * args.outer_iterations

    loss_valid_model_train_new = loss_valid_model_train.clone()
    if args.dataset == 'stsb':
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    for i, (tokens, labels, indices) in enumerate(tqdm(train_loader)):
        # Pre-processing
        tokens, labels = tokens.to(device), labels.to(device)
        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)

        if args.no_diff:
            y_pred_diff_batch = torch.zeros_like(torch.Tensor(y_pred_diff[idx_map[indices].numpy()])).to(device)
        else:
            y_pred_diff_batch = torch.Tensor(y_pred_diff[idx_map[indices].numpy()]).to(device)
        y_train_onehot_batch = torch.Tensor(y_train_onehot[idx_map[indices].numpy()]).to(device)
        measurement_batch = torch.Tensor(measurement[idx_map[indices].numpy()]).to(device)

        # Data valuation
        logit, penul = model(tokens, get_penul=True)

        if args.measurement:
            input_estim = torch.cat([measurement_batch, y_train_onehot_batch], dim=-1)
        else:
            input_estim = torch.cat([penul, y_train_onehot_batch], dim=-1)  # (B, dim + class)
            ## Debugging
            ## input_estim = torch.zeros(input_estim.size()).to(device)

        if args.normalize:
            input_estim = standard_normalization(input_estim)
            y_pred_diff_batch = standard_normalization(y_pred_diff_batch)

        ##### Inference of value estimator
        est_dv_curr = estimator(input_estim, y_pred_diff_batch)

        ##### Sample the mask
        sel_prob_curr = np.random.binomial(1, est_dv_curr.data.cpu().numpy(), est_dv_curr.data.cpu().shape)

        # Update model with obtained sample weights
        select_mask = torch.Tensor(sel_prob_curr).to(device)
        loss = (select_mask * criterion(logit, labels)).mean()

        est_optimizer.zero_grad()
        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

        # Reward computation
        valid_perf = loss_valid_model_train
        pred_valid, loss_valid, indices_partial = predicts_partial(args, val_loader, model)

        valid_perf_partial = valid_perf[indices_partial]
        dvrl_perf = loss_valid

        if args.base:
            reward_curr = (dvrl_perf - valid_perf_partial).mean().to(device)
        else:
            reward_curr = (dvrl_perf).mean().to(device)

        # Update the baseline
        loss_valid_model_train_new[indices_partial] *= ((args.window_size - 1) / args.window_size)
        loss_valid_model_train_new[indices_partial] += ((1 / args.window_size) * loss_valid)

        # Generator loss (REINFORCE algorithm)
        sel_prob_curr = torch.Tensor(sel_prob_curr).to(device)
        prob = torch.sum(
            sel_prob_curr * torch.log(est_dv_curr + eps) + (1 - sel_prob_curr) * torch.log(1 - est_dv_curr + eps))

        dve_loss = -1 * reward_curr * prob
        if args.no_regular:
            reg = torch.Tensor([0]).to(device)
        else:
            reg = args.lambda_reg * (torch.clamp(torch.mean(est_dv_curr) - threshold_u, min=0) + torch.clamp(
                (1 - threshold_u) - torch.mean(est_dv_curr), min=0))
            reg += args.lambda_reg * (torch.clamp(interval - torch.abs(torch.mean(est_dv_curr) - 0.5), min=0))
        # Update Estimator

        model_optimizer.zero_grad()
        est_optimizer.zero_grad()
        (dve_loss + reg).backward()
        est_optimizer.step()

        # Logging
        writer.add_scalars('rl optimization', {'reward_curr': reward_curr.mean().detach().cpu().item(),
                                               'dve_loss': dve_loss.mean().detach().cpu().item(),
                                               'reg': reg.mean().detach().cpu().item(),
                                               'est_dv_curr': est_dv_curr.mean().detach().cpu().item(),
                                               'dvrl_perf': dvrl_perf.mean().item(),
                                               'base_perf': valid_perf.mean().item()}, n_iter)

        n_iter += 1
        losses['est_dv_curr'].update(est_dv_curr.mean().detach().cpu().item(), len(dvrl_perf))
        losses['dv_max'].update(est_dv_curr.max().detach().cpu().item(), len(dvrl_perf))
        losses['dv_min'].update(est_dv_curr.min().detach().cpu().item(), len(dvrl_perf))

        losses['reward_curr'].update(reward_curr.mean().detach().cpu().item(), len(dvrl_perf))
        losses['dve_loss'].update(dve_loss.mean().detach().cpu().item(), len(dvrl_perf))
        losses['reg'].update(reg.mean().detach().cpu().item(), len(dvrl_perf))

    msg = '[est_dv %.3f] [max_dv %.3f] [min_dv %.3f] [reward %.3f] [loss_dve %.3f] [reg %.3f]' \
          % (losses['est_dv_curr'].average, losses['dv_max'].average, losses['dv_min'].average,
             losses['reward_curr'].average, losses['dve_loss'].average, losses['reg'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)

    return loss_valid_model_train_new

def predicts(args, loader, model, label=False):
    model.eval()

    all_preds, all_labels = [], []
    for _, (tokens, labels, _) in enumerate(iter(loader)):
        # Pre-processing
        # batch_size = tokens.size(0)
        # tokens, _ = cut_input(args, tokens)

        tokens = tokens.to(device)
        labels = labels.to(device)

        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)

        with torch.no_grad():
            logits = model(tokens)

        all_preds.append(logits.softmax(dim=-1).cpu())
        all_labels.append(labels.cpu())

    if label:
        return torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)
    else:
        return torch.cat(all_preds, dim=0)

def predicts_partial(args, loader, model):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    all_preds, all_losses, all_indices = [], [], []
    for i, (tokens, labels, indices) in enumerate(iter(loader)):
        tokens = tokens.to(device)
        labels = labels.to(device)

        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)

        with torch.no_grad():
            logits = model(tokens)

        all_preds.append(logits.cpu())
        all_losses.append(criterion(logits, labels).cpu())
        all_indices.append(indices)

        if i == args.n_batch_reward:
            break

    return torch.cat(all_preds, dim=0), torch.cat(all_losses, dim=0), torch.cat(all_indices, dim=0)


def train(args, train_tuples, model, optimizer):
    model.train()

    if args.dataset == 'stsb':
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    tokens, labels, select_mask = train_tuples[0], train_tuples[1], train_tuples[2]
    select_mask = torch.Tensor(select_mask[:, 0]).to(device)

    for _ in range(args.inner_iterations):
        batch_idx = np.random.permutation(len(tokens))[:args.batch_size]
        tokens_train, labels_train, select_mask_train = tokens[batch_idx], labels[batch_idx], select_mask[batch_idx]

        out_cls = model(tokens_train)

        loss = (select_mask_train * criterion(out_cls, labels_train)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def data_valuator(x_train, y_train):
    """Returns data values using the data valuator model.
    Args:
      x_train: training features (numpy)
      y_train: training labels (numpy)
    Returns:
      final_dat_value: final data values of the training samples
    """

    # One-hot encoded labels
    y_train_onehot = np.eye(len(np.unique(y_train)))[y_train.astype(int)] # N x K array
    y_train_valid_pred = self.val_model.predict_proba(x_train)

    # Generates y_train_hat
    y_train_hat = np.abs(y_train_onehot - y_train_valid_pred)

    # Estimates data value
    est_data_value = self.data_value_evaluator()

    return est_data_value
