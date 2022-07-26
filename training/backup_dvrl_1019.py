import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from training.common import AverageMeter, one_hot, cut_input, get_embed, data_aug, sym_kld
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = 1e-8

def standard_normalization(features):
    mean = features.mean(dim=-1, keepdim=True).detach()
    std = features.std(dim=-1, keepdim=True).detach()
    return (features - mean) / (std + eps)

def train_dvrl(args, train_loader, idx_map, val_loader, model, estimator, model_optimizer, est_optimizer,
               pred_valid_model_train, y_pred_diff, y_train_onehot, y_valid, measurement, epoch=0, logger=None, writer=None):
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
    losses['reward_curr'] = AverageMeter()
    losses['acc_dvrl'] = AverageMeter()

    n_iter = (epoch - 1) * args.outer_iterations

    pred_valid_model_train_new = pred_valid_model_train.clone()

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

        if args.logit:
            input_estim = torch.cat([logit, y_train_onehot_batch], dim=-1)  # (B, dim + class)
        elif args.measurement:
            input_estim = torch.cat([measurement_batch, y_train_onehot_batch], dim=-1)
        else:
            input_estim = torch.cat([penul, y_train_onehot_batch], dim=-1) # (B, dim + class)

        input_estim = standard_normalization(input_estim)
        y_pred_diff_batch = standard_normalization(y_pred_diff_batch)

        ##### Inference of value estimator
        est_dv_curr = estimator(input_estim, y_pred_diff_batch)
        
        ##### Sample the mask
        sel_prob_curr = np.random.binomial(1, est_dv_curr.data.cpu().numpy(), est_dv_curr.data.cpu().shape)

        # Update model with obtained sample weights
        train(args, [tokens, labels, sel_prob_curr], model, model_optimizer)

        # Reward computation
        valid_perf = (pred_valid_model_train_new.max(dim=-1)[1] == y_valid).float()

        if args.n_batch_reward > 0:
            pred_valid, indices_partial = predicts_partial(args, val_loader, model)
            y_valid_partial = y_valid[indices_partial]

            valid_perf_partial = valid_perf[indices_partial]
            dvrl_perf = (pred_valid.max(dim=-1)[1] == y_valid_partial).float()
            dvrl_acc = dvrl_perf.detach().cpu().sum() / len(pred_valid)

            reward_curr = (dvrl_perf - valid_perf_partial).sum().to(device)

            # Update the baseline
            pred_valid_model_train_new[indices_partial] *= ((args.window_size - 1) / args.window_size)
            pred_valid_model_train_new[indices_partial] += ((1 / args.window_size)  * pred_valid)
        else:
            pred_valid = predicts(args, val_loader, model, label=False)
            dvrl_perf = (pred_valid.max(dim=-1)[1] == y_valid).float()
            dvrl_acc = dvrl_perf.detach().cpu().sum() / len(dvrl_perf)

            ##### Validation accuracy gain from re-weighting scheme
            ########## Q. As a baseline, the released implementation does not utilize the moving window... => need to be added
            reward_curr = (dvrl_perf - valid_perf).sum().to(device)

        # Generator loss (REINFORCE algorithm)
        sel_prob_curr = torch.Tensor(sel_prob_curr).to(device)
        prob = torch.sum(sel_prob_curr * torch.log(est_dv_curr + eps) + (1 - sel_prob_curr) * torch.log(1 - est_dv_curr + eps))

        print(reward_curr)
        print(prob)

        dve_loss = -1 * reward_curr * prob
        if args.no_regular:
            dve_loss = dve_loss
        else:
            dve_loss += args.lambda_reg * (torch.clamp(torch.mean(est_dv_curr) - threshold_u, min=0) + torch.clamp((1 - threshold_u) - torch.mean(est_dv_curr), min=0))
            dve_loss += args.lambda_reg * (torch.clamp(interval - torch.abs(torch.mean(est_dv_curr) - 0.5), min=0))

        # Update Estimator
        est_optimizer.zero_grad()
        dve_loss.backward()
        est_optimizer.step()

        # Logging
        writer.add_scalars('rl optimization', {'reward_curr': reward_curr.mean().detach().cpu().item(),
                                               'dve_loss': dve_loss.mean().detach().cpu().item(),
                                               'est_dv_curr': est_dv_curr.mean().detach().cpu().item(),
                                               'acc_dvrl': dvrl_acc.item()}, n_iter)

        n_iter += 1
        losses['est_dv_curr'].update(est_dv_curr.mean().detach().cpu().item(), len(dvrl_perf))
        losses['dv_max'].update(est_dv_curr.max().detach().cpu().item(), len(dvrl_perf))
        losses['dv_min'].update(est_dv_curr.min().detach().cpu().item(), len(dvrl_perf))

        losses['reward_curr'].update(reward_curr.mean().detach().cpu().item(), len(dvrl_perf))
        losses['acc_dvrl'].update(dvrl_acc.item(), len(dvrl_perf))
        losses['dve_loss'].update(dve_loss.mean().detach().cpu().item(), len(dvrl_perf))

    msg = '[est_dv %.3f] [max_dv %.3f] [min_dv %.3f] [reward %.3f] [loss_dve %.3f] [acc_val %.3f]' \
          % (losses['est_dv_curr'].average, losses['dv_max'].average, losses['dv_min'].average,
             losses['reward_curr'].average, losses['dve_loss'].average, losses['acc_dvrl'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)

    return pred_valid_model_train_new


def train_dvrl_epoch(args, train_loader, idx_map, val_loader, model, estimator, model_optimizer, est_optimizer,
               valid_perf, y_pred_diff, y_train_onehot, y_valid, epoch=0, logger=None, writer=None):
    # Encourages exploration
    threshold = 0.9

    model.train()
    estimator.train()

    losses = dict()
    losses['est_dv_curr'] = AverageMeter()
    losses['dv_max'] = AverageMeter()
    losses['dv_min'] = AverageMeter()

    losses['dve_loss'] = AverageMeter()
    losses['reward_curr'] = AverageMeter()
    losses['acc_dvrl'] = AverageMeter()

    if args.dataset == 'stsb':
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    train_iter = iter(train_loader)
    n_iter = (epoch - 1) * len(train_iter)
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

        # Data valuation
        logit, penul = model(tokens, get_penul=True)

        if args.logit:
            input_estim = torch.cat([logit.clone().detach(), y_train_onehot_batch], dim=-1)  # (B, dim + class)
        else:
            input_estim = torch.cat([penul.clone().detach(), y_train_onehot_batch], dim=-1)  # (B, dim + class)

        input_estim = standard_normalization(input_estim)
        y_pred_diff_batch = standard_normalization(y_pred_diff_batch)

        ##### Inference of value estimator
        est_dv_curr = estimator(input_estim, y_pred_diff_batch)

        ##### Sample the mask
        sel_prob_curr = np.random.binomial(1, est_dv_curr.data.cpu().numpy(), est_dv_curr.data.cpu().shape)

        ##### Exception (When selection probability is 0)
        if np.sum(sel_prob_curr) == 0:
            est_dv_curr = 0.5 * np.ones(np.shape(est_dv_curr))
            sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)

        # Update model with obtained sample weights
        select_mask = torch.Tensor(sel_prob_curr).to(device)
        loss = (select_mask * criterion(logit, labels)).mean()

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

        # Reward computation
        if args.n_batch_reward > 0:
            pred_valid, indices_partial = predicts_partial(args, val_loader, model)
            y_valid_partial = y_valid[indices_partial]

            valid_perf_partial = valid_perf[indices_partial]
            dvrl_perf = (pred_valid.max(dim=-1)[1] == y_valid_partial).float()
            dvrl_acc = dvrl_perf.detach().cpu().sum() / len(pred_valid)

            reward_curr = (dvrl_perf - valid_perf_partial).sum().to(device)
        else:
            pred_valid = predicts(args, val_loader, model, label=False)
            dvrl_perf = (pred_valid.max(dim=-1)[1] == y_valid).float()
            dvrl_acc = dvrl_perf.detach().cpu().sum() / len(dvrl_perf)

            ##### Validation accuracy gain from re-weighting scheme
            ########## Q. As a baseline, the released implementation does not utilize the moving window... => need to be added
            reward_curr = (dvrl_perf - valid_perf).sum().to(device)

        # Generator loss (REINFORCE algorithm)
        sel_prob_curr = torch.Tensor(sel_prob_curr).to(device)
        prob = torch.sum(
            sel_prob_curr * torch.log(est_dv_curr + eps) + (1 - sel_prob_curr) * torch.log(1 - est_dv_curr + eps))
        dve_loss = -1 * reward_curr * prob
        dve_loss += 1000 * (torch.clamp(torch.mean(est_dv_curr) - threshold, min=0)
                            + torch.clamp((1 - threshold) - torch.mean(est_dv_curr), min=0))

        # Update Estimator
        est_optimizer.zero_grad()
        dve_loss.backward()
        est_optimizer.step()

        # Logging
        writer.add_scalars('rl optimization', {'reward_curr': reward_curr.mean().detach().cpu().item(),
                                               'dve_loss': dve_loss.mean().detach().cpu().item(),
                                               'est_dv_curr': est_dv_curr.mean().detach().cpu().item(),
                                               'acc_dvrl': dvrl_acc.item()}, n_iter)
        n_iter += 1
        losses['est_dv_curr'].update(est_dv_curr.mean().detach().cpu().item(), len(dvrl_perf))
        losses['dv_max'].update(est_dv_curr.max().detach().cpu().item(), len(dvrl_perf))
        losses['dv_min'].update(est_dv_curr.min().detach().cpu().item(), len(dvrl_perf))

        losses['reward_curr'].update(reward_curr.mean().detach().cpu().item(), len(dvrl_perf))
        losses['acc_dvrl'].update(dvrl_acc.item(), len(dvrl_perf))
        losses['dve_loss'].update(dve_loss.mean().detach().cpu().item(), len(dvrl_perf))

    msg = '[est_dv %.3f] [max_dv %.3f] [min_dv %.3f] [reward %.3f] [loss_dve %.3f] [acc_val %.3f]' \
          % (losses['est_dv_curr'].average, losses['dv_max'].average, losses['dv_min'].average,
             losses['reward_curr'].average, losses['dve_loss'].average, losses['acc_dvrl'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)

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

    all_preds, all_indices = [], []
    for i, (tokens, labels, indices) in enumerate(iter(loader)):
        tokens = tokens.to(device)

        with torch.no_grad():
            logits = model(tokens)

        all_preds.append(logits.softmax(dim=-1).cpu())
        all_indices.append(indices)

        if i == args.n_batch_reward:
            break

    return torch.cat(all_preds, dim=0), torch.cat(all_indices, dim=0)


def train(args, train_tuples, model, optimizer):
    model.train()

    if args.dataset == 'stsb':
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    tokens, labels, select_mask = train_tuples[0], train_tuples[1], train_tuples[2]
    select_mask = torch.Tensor(select_mask).to(device)

    for _ in range(args.inner_iterations):
        batch_idx = np.random.permutation(len(tokens))[:args.batch_size]
        tokens_train, labels_train = tokens[batch_idx], labels[batch_idx]

        out_cls = model(tokens_train)
        loss = (select_mask * criterion(out_cls, labels_train)).mean()
        #loss = criterion(out_cls, labels_train).mean()

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