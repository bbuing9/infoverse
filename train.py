import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import datetime
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from tqdm import tqdm

from eval import test_acc
from data import get_base_dataset
from models import load_backbone, Classifier
from common import CKPT_PATH, parse_args
from utils import Logger, set_seed, set_model_path, save_model, add_mislabel_dataset, pruning_dataset, cut_input, AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = parse_args(mode='train')

    ##### Set seed
    set_seed(args)

    ##### Set logs
    # Data pruning
    if args.data_ratio < 1.0 and args.selected is not None:
        args.train_type = args.train_type + "_" + args.selected
    log_name = f"{args.dataset}_R{args.data_ratio}_{args.backbone}_{args.train_type}_GA{args.grad_accumulation}_N{args.num_sub}_S{args.seed}"

    logger = Logger(log_name)
    log_dir = logger.logdir
    logger.log('Log_name =====> {}'.format(log_name))

    ##### Load models and dataset
    logger.log('Loading pre-trained backbone network... ({})'.format(args.backbone))
    backbone, tokenizer = load_backbone(args.backbone)

    logger.log('Initializing dataset...')
    dataset, train_loader, val_loader, test_loader = get_base_dataset(args.dataset, tokenizer, args.batch_size, args.seed)

    if args.data_ratio < 1.0 or (args.noisy_label_criteria is not None or args.noisy_label_path is not None):
        train_loader = set_new_loader(args, dataset)        

    logger.log('Initializing model and optimizer...')
    if args.dataset == 'wino':
        dataset.n_classes = 1
    model = Classifier(args.backbone, backbone, dataset.n_classes, args.train_type).to(device)

    if args.pre_ckpt is not None:
        logger.log('Loading from pre-trained model')
        model.load_state_dict(torch.load(args.pre_ckpt))

    # Set optimizer (1) fixed learning rate and (2) no weight decay
    optimizer = optim.Adam(model.parameters(), lr=args.model_lr, weight_decay=0)
    t_total = len(train_loader) * args.epochs
    
    logger.log('Lr schedule: Linear')
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(0.06 * t_total), num_training_steps=t_total) 
    
    logger.log('==========> Start training ({})'.format(args.train_type))
    best_acc, final_acc = 0, 0
    
    for epoch in range(1, args.epochs + 1):
        train_base(args, train_loader, model, optimizer, scheduler, epoch, logger)
        best_acc, final_acc = eval_func(args, model, val_loader, test_loader, logger, best_acc, final_acc)

        # Save model
        if args.save_ckpt:
            logger.log('Save model...')
            save_model(args, model, log_dir, dataset, epoch)
   
    logger.log('================>>>>>> Final Test Accuracy: {}'.format(final_acc))

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
        labels = labels.to(device).squeeze(1)

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

    
def eval_func(args, model, val_loader, test_loader, logger, best_acc, final_acc):
    # other_metric; [mcc, f1, p, s]
    acc, other_metric = test_acc(args, val_loader, model, logger)

    if args.dataset == 'cola':
        metric = other_metric[0]
    elif args.dataset == 'stsb':
        metric = other_metric[2]
    else:
        metric = acc

    if metric >= best_acc:
        # As val_data == test_data in GLUE, do not inference it again.
        if args.dataset == 'wnli' or args.dataset == 'rte' or args.dataset == 'mrpc' or args.dataset == 'stsb' or \
                args.dataset == 'cola' or args.dataset == 'sst2' or args.dataset == 'qnli' or args.dataset == 'qqp':
            t_acc, t_other_metric = acc, other_metric
        else:
            t_acc, t_other_metric = test_acc(args, test_loader, model, logger)

        if args.dataset == 'cola':
            t_metric = t_other_metric[0]
        elif args.dataset == 'stsb':
            t_metric = t_other_metric[2]
        else:
            t_metric = t_acc

        # Update test accuracy based on validation performance
        best_acc, final_acc = metric, t_metric

        if args.dataset == 'mrpc' or args.dataset == 'qqp':
            logger.log('========== Test Acc/F1 ==========')
            logger.log('Test acc: {:.3f} Test F1: {:.3f}'.format(final_acc, t_other_metric[1]))
        elif args.dataset == 'stsb':
            logger.log('========== Test P/S ==========')
            logger.log('Test P: {:.3f} Test S: {:.3f}'.format(t_other_metric[2], t_other_metric[3]))
        elif args.dataset == 'mnli':
            logger.log('========== Test m/mm ==========')
            logger.log('Test matched/mismatched: {:.3f}/{:.3f}'.format(best_acc, final_acc))
        else:
            logger.log('========== Val Acc ==========')
            logger.log('Val acc: {:.3f}'.format(best_acc))
            logger.log('========== Test Acc ==========')
            logger.log('Test acc: {:.3f}'.format(final_acc))

    return best_acc, final_acc

def set_new_loader(args, dataset):
    # Adding syntactic noise label to train dataset
    if args.noisy_label_criteria is not None or args.noisy_label_path is not None:
        train_loader = add_mislabel_dataset(args, dataset.train_dataset, dataset.class_idx)
    else:
        n_sub = int(args.data_ratio * len(dataset.train_dataset))
        data, labels, idxs = dataset.train_dataset[:][0], dataset.train_dataset[:][1], dataset.train_dataset[:][2]
        measures = np.load("./gen_info/{}_{}_infoverse.npy".format(args.dataset, args.backbone))

        if args.selected_idx is not None:
            indices = np.load(args.selected_idx)
            if len(indices.shape) == 2:
                maps = {'0.83': 0, '0.66': 1, '0.5': 2, '0.33': 3, '0.25': 4, '0.17': 5, '0.13': 6, '0.09': 7, '0.05': 8}
                sel_idx = indices[maps[str(args.data_ratio)]][:n_sub]
            else:
                sel_idx = indices[:n_sub]
        elif args.selected == 'easy':
            scores = measures[:, 0]
            sel_idx = np.argsort(scores)[::-1][:n_sub]
            sel_idx = np.array(sel_idx)
        elif args.selected == 'hard':
            scores = measures[:, 0]
            sel_idx = np.argsort(scores)[:n_sub]
            sel_idx = np.array(sel_idx)
        elif args.selected == 'ambig':
            scores = measures[:, 1]
            sel_idx = np.argsort(scores)[::-1][:n_sub]
            sel_idx = np.array(sel_idx)
        elif args.selected == 'ent':
            scores = measures[:, 5]
            sel_idx = np.argsort(scores)[::-1][:n_sub]
            sel_idx = np.array(sel_idx)
        elif args.selected == 'dens':
            scores = measures[:, 21]
            sel_idx = np.argsort(scores)[::-1][:n_sub]
            sel_idx = np.array(sel_idx)
        else:
            sorted_idx = torch.randperm(len(data))
            sel_idx = sorted_idx[:n_sub]

        sub_dataset = TensorDataset(data[sel_idx, :], labels[sel_idx, :], idxs[sel_idx])
        train_loader = DataLoader(sub_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size, num_workers=4)
    
    return train_loader

if __name__ == "__main__":
    main()
