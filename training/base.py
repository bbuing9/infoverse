import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from training.common import AverageMeter, one_hot, cut_input, get_embed, data_aug, sym_kld
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train_base(args, loader, model, optimizer, scheduler, epoch=0, logger=None):
    model.train()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['cls_acc'] = AverageMeter()

    if args.dataset == 'stsb':
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    steps = epoch * len(loader)
    for i, (tokens, labels, _) in enumerate(tqdm(loader)):
        steps += 1
        batch_size = tokens.size(0)
        if args.dataset == 'wino':
            tokens = tokens[:, 0, :, :]
            labels = labels - 1
        else:
            tokens, _ = cut_input(args, tokens)

        tokens = tokens.to(device)
        labels = labels.to(device)

        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)

        out_cls = model(tokens)
        loss = criterion(out_cls, labels).mean()
        (loss / args.grad_accumulation).backward()
        scheduler.step()
        if steps % args.grad_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

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
