import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from meta_models import build_model
from training.common import AverageMeter, one_hot, cut_input, get_embed, data_aug, sym_kld
from tqdm import tqdm
from magic_module import MagicModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_meta_eff(args, measurements, train_penuls, val_penuls, train_labels, val_labels, labels_mix, linear_model, optimizer, value_model, meta_optimizer, epoch=0, logger=None):
    # Codes are mainly adopted from https://github.com/xjtushujun/meta-weight-net
    train_labels = train_labels.to(device).long()
    val_labels = val_labels.to(device).long()

    for batch_idx in range(args.inner_iterations):
        ##### 1. One-step update for training samples
        # Re-initialization
        meta_linear = build_model(train_penuls.size(-1), 3, False)
        meta_linear.load_state_dict(linear_model.state_dict())

        outputs = meta_linear(train_penuls)
        cost_v = F.cross_entropy(outputs, train_labels, reduce=False).unsqueeze(-1)

        # Valuation using value model
        #v_lambda = value_model(cost_v.data)
        v_lambda = value_model(measurements[:, :])
        l_f_meta = torch.sum(cost_v * v_lambda) / (1e-8 + len(cost_v) * v_lambda.mean())

        # One-step update
        meta_linear.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_linear.params()), create_graph=True)
        meta_linear.update_params(lr_inner=args.model_lr, source_params=grads)
        del grads

        ##### 2. Validation loss minimization
        y_g_hat = meta_linear(val_penuls)
        l_g_meta = F.cross_entropy(y_g_hat, val_labels)

        meta_optimizer.zero_grad()
        l_g_meta.backward()
        meta_optimizer.step()

        # logging
        pred_meta = y_g_hat.max(dim=-1)[1]
        acc_meta = (pred_meta == val_labels.data).float().sum() / len(pred_meta)

        ##### 3. Weighted loss minimization with training samples
        outputs = linear_model(train_penuls)
        cost_w = F.cross_entropy(outputs, train_labels, reduce=False)
        cost_w = torch.reshape(cost_w, (len(cost_w), 1))

        with torch.no_grad():
            #w_new = value_model(cost_w)
            w_new = value_model(measurements[:, :])
        loss = torch.sum(cost_w * w_new) / (1e-8 + len(cost_w) * w_new.mean())
        #loss = cost_w.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_train = outputs.max(dim=-1)[1]
        acc_train = (pred_train == train_labels.data).float().sum() / len(pred_train)

        train_loss = loss.item()
        meta_loss = l_g_meta.item()
        w_train = w_new.mean().item()
        w_max = w_new.max().item()
        w_min = w_new.min().item()

        w_clean = w_new[labels_mix == 1].mean().item()
        w_noisy = w_new[labels_mix == 0].mean().item()

        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'W_net: %.3f (max: %.3f, min:%.3f)\t'
                  'v_clean: %.3f, v_noisy: %.3f\t'
                  'Prec@1 %.3f\t'
                  'Prec_meta@1 %.3f' % (epoch, args.epochs, batch_idx + 1, len(train_penuls)/args.batch_size,
                      (train_loss), (meta_loss), (w_train), (w_max),  (w_min), w_clean, w_noisy, acc_train, acc_meta))

def train_meta(args, loader, val_loader, backbone, linear_model, optimizer, value_model, meta_optimizer, epoch=0, logger=None):
    # Codes are mainly adopted from https://github.com/xjtushujun/meta-weight-net
    backbone.eval()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['cls_acc'] = AverageMeter()

    val_loader_iter = iter(val_loader)
    train_loss = 0
    meta_loss = 0
    w_train, w_max, w_min = 0, 0, 0

    for batch_idx, (tokens, labels, _) in enumerate(tqdm(loader)):
        ##### 0. Pre-processing
        batch_size = tokens.size(0)
        tokens, _ = cut_input(args, tokens)

        tokens = tokens.to(device)
        labels = labels.to(device)

        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)

        with torch.no_grad():
            _, penuls = backbone(tokens, get_penul=True)

        ##### 1. One-step update for training samples
        # Re-initialization
        meta_linear = build_model(penuls.size(-1), 3)
        meta_linear.load_state_dict(linear_model.state_dict())

        outputs = meta_linear(penuls)
        cost_v = F.cross_entropy(outputs, labels, reduce=False).unsqueeze(-1)

        # Valuation using value model
        v_lambda = value_model(cost_v.data)
        l_f_meta = torch.sum(cost_v * v_lambda) / len(cost_v)

        # One-step update
        meta_linear.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_linear.params()), create_graph=True)
        meta_linear.update_params(lr_inner=args.model_lr, source_params=grads)
        del grads

        ##### 2. Validation loss minimization
        try:
            inputs_val, targets_val, _ = next(val_loader_iter)
        except StopIteration:
            val_loader_iter = iter(val_loader)
            inputs_val, targets_val, _ = next(val_loader_iter)

        inputs_val, targets_val = inputs_val.to(device), targets_val.squeeze(1).to(device)

        with torch.no_grad():
            _, v_penuls = backbone(inputs_val, get_penul=True)

        y_g_hat = meta_linear(v_penuls)
        l_g_meta = F.cross_entropy(y_g_hat, targets_val)

        meta_optimizer.zero_grad()
        l_g_meta.backward()
        meta_optimizer.step()

        # logging
        pred_meta = y_g_hat.max(dim=-1)[1]
        acc_meta = (pred_meta == targets_val.data).float().sum() / len(pred_meta)

        ##### 3. Weighted loss minimization with training samples
        outputs = linear_model(penuls)
        cost_w = F.cross_entropy(outputs, labels, reduce=False)
        cost_w = torch.reshape(cost_w, (len(cost_w), 1))

        with torch.no_grad():
            w_new = value_model(cost_w)
        loss = torch.sum(cost_w * w_new) / len(cost_w)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_train = outputs.max(dim=-1)[1]
        acc_train = (pred_train == labels.data).float().sum() / len(pred_train)

        train_loss += loss.item()
        meta_loss += l_g_meta.item()
        w_train += w_new.mean().item()
        w_max += w_new.max().item()
        w_min += w_new.min().item()

        losses['cls'].update(loss.item(), batch_size)
        losses['cls_acc'].update(acc_train.item(), batch_size)

        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'W_net: %.3f\t (max: %.3f, min:%.3f)'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % ((epoch + 1), args.epochs, batch_idx + 1, len(loader.dataset)/args.batch_size,
                      (train_loss / (batch_idx + 1)), (meta_loss / (batch_idx + 1)), (w_train / (batch_idx + 1)),
                                        (w_max / (batch_idx + 1)),  (w_min / (batch_idx + 1)), acc_train, acc_meta))

    msg = '[Epoch %2d] [AccC %.3f] [LossC %.3f]' % (epoch, losses['cls_acc'].average, losses['cls'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)

def train_meta_l2m(args, loader, val_loader, idx_map, measurements, model, optimizer, value_model, meta_optimizer, epoch=0, logger=None):
    # Codes are mainly adopted from https://github.com/xjtushujun/meta-weight-net
    model.train()
    value_model.train()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['cls_acc'] = AverageMeter()

    val_loader_iter = iter(val_loader)
    train_loss = 0
    meta_loss = 0
    w_train, w_max, w_min = 0, 0, 0

    for batch_idx, (tokens, labels, indices) in enumerate(tqdm(loader)):
        ##### 0. Pre-processing
        batch_size = tokens.size(0)
        tokens, _ = cut_input(args, tokens)

        tokens = tokens.to(device)
        labels = labels.to(device)

        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)

        if batch_idx > 0:
            ##### 1. One-step update for training samples
            magic_model = MagicModule(model)
            weights = value_model(measurements[idx_map[indices], :])
            weights_max = weights.data.max()

            # Q. should it be sample-wise?
            for i in range(batch_size):
                model.zero_grad()
                logits = model(tokens[i:i+1])
                loss = F.cross_entropy(logits, labels[i:i+1])
                grads = torch.autograd.grad(loss, [param for name, param in model.named_parameters()], allow_unused=True)
                grads = {param: grads[j] for j, (name, param) in enumerate(model.named_parameters())}

                deltas = _adam_delta(optimizer, model, grads)
                deltas = {name: (weights[i] / (1e-8 + weights_max)) * delta.data for name, delta in deltas.items()}
                magic_model.update_params(deltas)

            ##### 2. Validation loss minimization
            try:
                inputs_val, targets_val, _ = next(val_loader_iter)
            except StopIteration:
                val_loader_iter = iter(val_loader)
                inputs_val, targets_val, _ = next(val_loader_iter)

            inputs_val, targets_val = inputs_val.to(device), targets_val.squeeze(1).to(device)
            y_g_hat = magic_model(inputs_val)
            l_g_meta = F.cross_entropy(y_g_hat, targets_val)
            meta_optimizer.zero_grad()
            l_g_meta.backward()
            meta_optimizer.step()

            # logging
            pred_meta = y_g_hat.max(dim=-1)[1]
            acc_meta = (pred_meta == targets_val.data).float().sum() / len(pred_meta)

        ##### (ok) 3. Weighted loss minimization with training samples
        outputs = model(tokens)
        cost_w = F.cross_entropy(outputs, labels, reduce=False)
        cost_w = torch.reshape(cost_w, (len(cost_w), 1))

        with torch.no_grad():
            w_new = value_model(measurements[idx_map[indices], :])
        w_new /= (1e-8 + w_new.data.max())
        loss = torch.sum(cost_w * w_new) / len(cost_w)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_train = outputs.max(dim=-1)[1]
        acc_train = (pred_train == labels.data).float().sum() / len(pred_train)

        train_loss += loss.item()
        if batch_idx > 0:
            meta_loss += l_g_meta.item()
        w_train += w_new.mean().item()
        w_max += w_new.max().item()
        w_min += w_new.min().item()

        losses['cls'].update(loss.item(), batch_size)
        losses['cls_acc'].update(acc_train.item(), batch_size)

        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'W_net: %.3f\t (max: %.3f, min:%.3f)'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % ((epoch + 1), args.epochs, batch_idx + 1, len(loader.dataset)/args.batch_size,
                      (train_loss / (batch_idx + 1)), (meta_loss / (batch_idx + 1)), (w_train / (batch_idx + 1)),
                                        (w_max / (batch_idx + 1)),  (w_min / (batch_idx + 1)), acc_train, acc_meta))

    msg = '[Epoch %2d] [AccC %.3f] [LossC %.3f]' % (epoch, losses['cls_acc'].average, losses['cls'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)

def _adam_delta(optimizer, model, grads):
    deltas = {}
    for group in optimizer.param_groups:
        for param in group['params']:
            grad = grads[param]
            state = optimizer.state[param]
            if grad is None:
                continue

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']

            step = state['step'] + 1

            if group['weight_decay'] != 0:
                grad = grad + group['weight_decay'] * param.data

            exp_avg = exp_avg * beta1 + (1. - beta1) * grad
            exp_avg_sq = exp_avg_sq * beta2 + (1. - beta2) * grad * grad
            denom = exp_avg_sq.sqrt() + group['eps']

            bias_correction1 = 1. - beta1 ** step
            bias_correction2 = 1. - beta2 ** step
            step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

            deltas[param] = -step_size * exp_avg / denom

    param_to_name = {param: name for name, param in model.named_parameters()}

    return {param_to_name[param]: delta for param, delta in deltas.items()}

def predict(loader, model, label=False):

    all_preds, all_labels = [], []
    for _, (tokens, labels, _) in enumerate(iter(loader)):
        # Pre-processing
        batch_size = tokens.size(0)
        tokens, _ = cut_input(args, tokens)

        tokens = tokens.to(device)
        labels = labels.to(device)

        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)

        with torch.no_grad():
            logit = model(tokens)
        preds = torch.max(logit, dim=-1)[1].cpu()

        all_preds.append(preds)
        all_labels.append(labels.cpu())

    if label:
        return torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)
    else:
        return torch.cat(all_preds, dim=0)

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

def train_base_abl(args, loader, model, lin_model, optimizer, epoch=0, logger=None):
    model.train()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['cls_acc'] = AverageMeter()

    if args.dataset == 'stsb':
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    for i, (tokens, labels, _) in enumerate(tqdm(loader)):
        batch_size = tokens.size(0)
        tokens, _ = cut_input(args, tokens)

        tokens = tokens.to(device)
        labels = labels.to(device)

        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)

        with torch.no_grad():
            _, penuls = model(tokens, get_penul=True)
        out_cls = lin_model(penuls)

        loss = criterion(out_cls, labels).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # cls_acc
        _, pred_cls = out_cls.max(dim=1)
        corrects = (pred_cls == labels).float()
        acc_cls = corrects.sum() / batch_size

        losses['cls'].update(loss.item(), batch_size)
        losses['cls_acc'].update(acc_cls.item(), batch_size)

    msg = '[Epoch %2d] [AccC %.3f] [LossC %.3f]' % (epoch, losses['cls_acc'].average, losses['cls'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)

def train_aug(args, loader, model, optimizer, aug_src, epoch=0, logger=None):
    model.train()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['cls_acc'] = AverageMeter()
    losses['aug'] = AverageMeter()
    losses['aug_acc'] = AverageMeter()

    if args.dataset == 'stsb':
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    for i, (tokens, labels, indices) in enumerate(tqdm(loader)):
        batch_size = tokens.size(0)

        tokens = tokens.to(device)
        labels = labels.to(device)
        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)

        # Augmentation
        aug_tokens = aug_src[indices]
        tokens, embed, aug_tokens, aug_embed = data_aug(args, model, tokens, labels, aug_tokens)

        out_cls = model(tokens, inputs_embed=None)
        out_aug = model(aug_tokens, inputs_embed=aug_embed)

        # Total loss
        loss_cls = criterion(out_cls, labels).mean()
        loss_aug = criterion(out_aug, labels).mean()
        loss_symkld = sym_kld(out_aug, out_cls).mean()

        loss = loss_cls + args.lambda_aug * loss_aug + args.lambda_kl * loss_symkld

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        _, pred_cls = out_cls.max(dim=1)
        corrects = (pred_cls == labels).float()
        acc_cls = corrects.sum() / batch_size

        _, pred_aug = out_aug.max(dim=1)
        corrects_aug = (pred_aug == labels).float()
        acc_aug = corrects_aug.sum() / batch_size

        losses['cls'].update(loss_cls.item(), batch_size)
        losses['cls_acc'].update(acc_cls.item(), batch_size)
        losses['aug'].update(loss_aug.item(), batch_size)
        losses['aug_acc'].update(acc_aug.item(), batch_size)

    msg = '[Epoch %2d] [Accuracy Orig %.3f] [Accuracy Aug %.3f] [Loss Orig %.3f] [Loss Aug %.3f]' \
          % (epoch, losses['cls_acc'].average, losses['aug_acc'].average,  losses['cls'].average, losses['aug'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)

def train_mixup(args, loader, model, optimizer, epoch=0, logger=None):
    model.train()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['cls_acc'] = AverageMeter()

    for i, (tokens, labels, indices) in enumerate(tqdm(loader)):
        batch_size = tokens.size(0)
        tokens, attention_mask = cut_input(args, tokens)

        tokens = tokens.to(device)
        labels = labels.to(device)
        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)
        embed = get_embed(args, model, tokens)

        # Mixup
        if args.mixup_alpha == 0:
            l = 1
        else:
            l = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            l = max(l, 1 - l)
        idx = torch.randperm(embed.size(0))
        labels_a, labels_b = labels, labels[idx]

        embed_a, embed_b = embed, embed[idx]
        mixed_embed = l * embed_a + (1 - l) * embed_b

        out_cls = model(tokens, inputs_embed=mixed_embed)  # (B, C)

        # mixed loss
        if args.dataset != 'stsb':
            loss = l * F.cross_entropy(out_cls, labels_a) + (1 - l) * F.cross_entropy(out_cls, labels_b)
        else:
            labels = l * labels_a.float() + (1 - l) * labels_b.float()
            loss = F.mse_loss(out_cls, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # cls_acc
        _, pred_cls = out_cls.max(dim=1)
        corrects = (pred_cls == labels_a).float()
        corrects2 = (pred_cls == labels_b).float()
        acc_cls = (corrects.sum() + corrects2.sum()) / (2 * batch_size)

        losses['cls'].update(loss.item(), batch_size)
        losses['cls_acc'].update(acc_cls.item(), batch_size)

    msg = '[Epoch %2d] [AccC %.3f] [LossC %.3f]' % (epoch, losses['cls_acc'].average, losses['cls'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)