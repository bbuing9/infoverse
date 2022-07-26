import os
import sys
import time
from datetime import datetime
import shutil
import math
import json

import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, Subset
from data.base_dataset import create_tensor_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""
    def __init__(self, fn):
        if not os.path.exists("./logs/"):
            os.mkdir("./logs/")

        logdir = 'logs/' + fn
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        if len(os.listdir(logdir)) != 0:
            # ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
            #                 "Will you proceed [y/N]? ")
            print("log_dir is not empty. original code shows input prompter, but hard-coding for convenience")
            ans = 'y' #TODO: remove it when doing commit or push
            if ans in ['y', 'Y']:
                shutil.rmtree(logdir)
            else:
                exit(1)
        self.set_dir(logdir)

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
        self.log_file.flush()

        print('[%s] %s' % (datetime.now(), string))
        sys.stdout.flush()

    def log_dirname(self, string):
        self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
        self.log_file.flush()

        print('%s (%s)' % (string, self.logdir))
        sys.stdout.flush()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def set_model_path(args, dataset, epoch, add_str=None):
    # Naming the saving model
    suffix = "_"
    suffix += str(args.train_type)
    suffix += "_epoch" + str(epoch)

    if add_str is not None:
        suffix += add_str

    return dataset.base_path + suffix + '.model'

def save_model(args, model, log_dir, dataset, epoch, add_str=None):
    # Save the model
    if isinstance(model, nn.DataParallel):
        model = model.module

    os.makedirs(log_dir, exist_ok=True)
    model_path = set_model_path(args, dataset, epoch, add_str)
    save_path = os.path.join(log_dir, model_path)
    torch.save(model.state_dict(), save_path)

def get_raw_data(args, dataset, tokenizer):
    temp_loader = DataLoader(dataset.train_dataset, shuffle=False, drop_last=False, batch_size=1, num_workers=4)

    train_data, train_labels = [], []
    for i, (tokens, labels, _) in enumerate(temp_loader):
        if args.backbone in ['bert', 'albert']:
            num_tokens = (tokens[0] > 0).sum()
        else:
            num_tokens = (tokens[0] != 1).sum()
        orig_sentence = tokenizer.decode(tokens[0, 1:num_tokens - 1])
        train_data.append(orig_sentence)
        train_labels.append(int(labels[0]))

    orig_src_loc = './pre_augment/' + args.dataset + '_' + args.backbone
    with open(orig_src_loc + '_data.txt', "w") as fp:
        json.dump(train_data, fp)

    with open(orig_src_loc + '_label.txt', "w") as fp:
        json.dump(train_labels, fp)

def augmenting(args, dataset, tokenizer, save_loc):
    # Generate text file for raw data
    print('========== Loading raw data ==========')
    get_raw_data(args, dataset, tokenizer)
    print('========== Constructing augmented samples ==========')
    generate_augments(args, tokenizer, save_loc)

def load_augment(args, dataset, tokenizer):
    aug_src_loc = './pre_augment/' + args.dataset + '_' + args.aug_type + '_' + args.backbone

    if not os.path.exists(aug_src_loc + '.npy'):
        print('Generating Augmented Samples')
        augmenting(args, dataset, tokenizer, aug_src_loc)

    aug_src = np.load(aug_src_loc + '.npy')

    return torch.LongTensor(aug_src)

def load_selected_aug(args, dataset):
    select_aug_data = np.load(args.selected + '_data.npy')
    select_aug_data = torch.LongTensor(select_aug_data)

    select_aug_label = np.load(args.selected + '_label.npy')
    select_aug_label = torch.LongTensor(select_aug_label).unsqueeze(1)

    select_aug_index = torch.arange(len(select_aug_data)) + len(dataset.train_dataset)
    select_dataset = TensorDataset(select_aug_data, select_aug_label, select_aug_index)
    concat_dataset = ConcatDataset([dataset.train_dataset, select_dataset])

    return DataLoader(concat_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size, num_workers=4)

def add_mislabel_dataset(args, datas, class_idx, infer=False):
    num_noisy_label = int(len(datas) * args.noisy_label_ratio)
    inputs, labels, idxs = [], [], []

    if args.noisy_label_path is not None:
        # index 0 for original,  1 for noisy label
        labels_info = np.load(args.noisy_label_path)
        labels_mix = (labels_info[:, 1] == labels_info[:, 0]).astype(int)

        for idx, data in enumerate(datas):
            inputs.append(data[0])
            idxs.append(data[2])
            if labels_mix[idx] == 0:
                labels.append(
                    torch.tensor(labels_info[idx,1]).long())
            else:
                labels.append(data[1][0])
    else:
        if args.noisy_label_criteria == 'random':
            all_idx = list(range(len(datas)))
            random.shuffle(all_idx)
            shuffled_idx = all_idx
            mislabeled_idx = shuffled_idx[:num_noisy_label]
            clean_labeled_idx = shuffled_idx[num_noisy_label:]

            assert len(shuffled_idx) == (len(mislabeled_idx) + len(clean_labeled_idx))
        else:
            criteria = np.load(
                './scores/{}_{}_{}_{}.npy'.format(args.dataset, args.data_ratio, args.backbone, args.noisy_label_criteria))

            sort_conf_order = np.argsort(criteria[::-1])  # descending

            mislabeled_idx = sort_conf_order[:num_noisy_label]
            clean_labeled_idx = sort_conf_order[num_noisy_label:]

            assert len(sort_conf_order) == (len(mislabeled_idx) + len(clean_labeled_idx))

        for idx, data in enumerate(datas):
            inputs.append(data[0])
            idxs.append(data[2])
            if idx in mislabeled_idx:
                data = list(data)
                label_list = class_idx.copy()
                label_list.remove(data[1]) # To randomly permute the label except the true one.
                labels.append(
                    torch.tensor(random.choice(label_list)).long())  # torch.tensor(random.choice(label_list)).long())
            else:
                labels.append(data[1][0])

        orig_labels = datas[:][1].numpy()
        np.save('./scores_noisy/{}_{}_{}_{}_{}_noisy_label.npy'.format(args.dataset, args.data_ratio, args.backbone, args.noisy_label_criteria, args.noisy_label_ratio),
                np.concatenate([orig_labels, np.array(labels).reshape(-1, 1)], axis=1)) # [orig_label, noisy_labels]

    mixed_dataset = create_tensor_dataset(inputs, labels, idxs)

    if infer:
        return DataLoader(mixed_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=4)
    else:
        return DataLoader(mixed_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size, num_workers=4)

def pruning_dataset(args, datas, infer=False):
    if args.pruning_sample_path is not None:
        pruned_idx = np.load('./pruning/{}_{}_{}_{}_pruning_idx.npy'.format(args.dataset, args.data_ratio, args.backbone, args.pruning_sample_ratio))
    else:
        all_idx = list(range(len(datas)))
        pruning_num = int(args.pruning_sample_ratio * len(datas))
        pruned_idx = random.sample(all_idx, pruning_num)
        remained_idx = list(set(all_idx) - set(pruned_idx))
        assert (len(pruned_idx) + len(remained_idx)) == len(all_idx)
    pruned_dataset = Subset(datas, np.array(remained_idx))
    if infer:
        return DataLoader(pruned_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=4)
    else:
        return DataLoader(pruned_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size, num_workers=4), pruned_idx

