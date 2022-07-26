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

from eval import test_acc
from data import get_base_dataset
from models import load_backbone, Classifier
from training import train_base, train_mixup, train_aug
from common import CKPT_PATH, parse_args
from utils import Logger, set_seed, set_model_path, save_model, load_augment, load_selected_aug, add_mislabel_dataset, \
    pruning_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = parse_args(mode='train')

    ##### Set seed
    set_seed(args)

    ##### Set logs
    log_name = f"{args.dataset}_R{args.data_ratio}_{args.backbone}_{args.train_type}_N{args.num_sub}_S{args.seed}"

    logger = Logger(log_name)
    log_dir = logger.logdir
    logger.log('Log_name =====> {}'.format(log_name))

    ##### Load models and dataset
    logger.log('Loading pre-trained backbone network... ({})'.format(args.backbone))
    backbone, tokenizer = load_backbone(args.backbone)

    logger.log('Initializing dataset...')
    dataset, train_loader, val_loader, test_loader = get_base_dataset(args.dataset, tokenizer, args.batch_size, args.data_ratio, args.seed)

    if args.num_sub < 1.1:
        if args.dataset == 'sst2':
            book_tokens = np.load('book_10000_tokens.npy')
            book_pseudo = np.load('book_10000_sst2_pseudo.npy')
            measures = np.load('0108_book_sst2_measures.npy')
        else:
            book_tokens = np.load('book_cola_tokens.npy')
            book_pseudo = np.load('book_10000_cola_pseudo.npy')
            measures = np.load('0109_book_cola_measures.npy')
        book_tokens = torch.Tensor(book_tokens).long()
        book_pseudo = torch.Tensor(book_pseudo).long()
        n_sub = int(args.num_sub * len(book_tokens))

        logger.log('Number of selected samples: {}...'.format(n_sub))

        data, labels, idxs = dataset.train_dataset[:]
        book_indices = len(idxs) + torch.arange(len(book_tokens))
        book_indices = book_indices.long()

        if args.selected_idx is not None:
            indices = np.load(args.selected_idx)
            if len(indices.shape) == 2:
                maps = {'0.83': 0, '0.66': 1, '0.5': 2, '0.33': 3, '0.25': 4, '0.17': 5, '0.13': 6, '0.09': 7, '0.05': 8}
                sel_idx = indices[maps[str(args.num_sub)]][:n_sub]
                print('here')
                print(len(sel_idx))
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
            sorted_idx = torch.randperm(len(book_tokens))
            sel_idx = sorted_idx[:n_sub]

        sub_data = book_tokens[sel_idx, :]
        sub_labels = book_pseudo[sel_idx, :]
        sub_idxs = book_indices[sel_idx]
        
        concat_data = torch.cat([data, sub_data], dim=0)
        concat_labels = torch.cat([labels, sub_labels], dim=0)
        concat_idxs = torch.cat([idxs, sub_idxs], dim=0)

        concat_dataset = TensorDataset(concat_data, concat_labels, concat_idxs)
        train_loader = DataLoader(concat_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size, num_workers=4)

    logger.log('Initializing model and optimizer...')
    if args.dataset == 'wino':
        dataset.n_classes = 1
    model = Classifier(args.backbone, backbone, dataset.n_classes, args.train_type).to(device)

    if args.pre_ckpt is not None:
        logger.log('Loading from pre-trained model')
        model.load_state_dict(torch.load(args.pre_ckpt))

    # Set optimizer (1) fixed learning rate and (2) no weight decay
    optimizer = optim.Adam(model.parameters(), lr=args.model_lr, weight_decay=0)
    #optimizer = AdamW(model.parameters(), lr=args.model_lr, eps=1e-8)
    t_total = len(train_loader) * args.epochs
    if args.linear:
        logger.log('Lr schedule: Linear')
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(0.06 * t_total), num_training_steps=t_total) 
    else:
        logger.log('Lr schedule: Constant')
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps = int(0.06 * t_total)) 

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    logger.log('==========> Start training ({})'.format(args.train_type))
    best_acc, final_acc = 0, 0
    aug_src =None
    
    for epoch in range(1, args.epochs + 1):
        #best_acc, final_acc = eval_func(args, model, val_loader, test_loader, logger, best_acc, final_acc)
        train_base(args, train_loader, model, optimizer, scheduler, epoch, logger)
        best_acc, final_acc = eval_func(args, model, val_loader, test_loader, logger, best_acc, final_acc)

        # Save model
        if args.save_ckpt:
            logger.log('Save model...')
            save_model(args, model, log_dir, dataset, epoch)
   
    logger.log('================>>>>>> Final Test Accuracy: {}'.format(final_acc))

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
        best_acc = metric
        final_acc = t_metric

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

if __name__ == "__main__":
    main()
