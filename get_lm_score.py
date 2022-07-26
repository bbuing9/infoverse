##
import os
import easydict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from models import load_backbone, Classifier

from data import get_base_dataset
from utils import Logger, set_seed, set_model_path, save_model, load_augment, add_mislabel_dataset
from training.common import cut_input, get_embed, data_aug

from scores_src import get_features, merge_multiple_models
from scores_src import avg_conf_variab, avg_forgetting, avg_aum
from scores_src import get_density_score, get_mlm_scores, masking_dataset, get_mlm_scores_jh, get_sentence_embedding
from scores_src import confidence, entropy, badge_grads_norm
from scores_src.ensembles import mc_dropout_models, el2n_score, ens_max_ent, ens_bald, ens_varR

args = easydict.EasyDict({"batch_size": 64,
                          "backbone": 'roberta_large',
                          "dataset": 'qnli',
                          "ood_dataset": 'trec',
                          "train_type": 'base',
                          "aug_type": 'none',
                          "seed": 1234,
                          "name": '1110_base',
                          "pre_ckpt": './logs/rte_R1.0_1110_base_S1234/rte_roberta-base_1110_base_epoch10.model',
                          "score_type": 'confidence',
                          "topK": True,
                          "data_ratio": 1.0,
                          "n_classes": 2,
                          "noisy_label_criteria": 'avg_conf',
                          "noisy_label_ratio": 0.0,
                        })

backbone, tokenizer = load_backbone(args.backbone)
dataset, train_loader, val_loader, test_loader = get_base_dataset(args.dataset, tokenizer, args.batch_size, args.data_ratio, args.seed, shuffle=False)
train_loader = DataLoader(dataset.train_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=4)
labels_v = dataset.val_dataset[:][1][:, 0].numpy()
labels_t = dataset.train_dataset[:][1][:, 0].numpy()
model = Classifier(args.backbone, backbone, dataset.n_classes, args.train_type).cuda()

mlm_score = get_mlm_scores_jh(args, model, train_loader)
np.save('qnli_mlm_score.npy', mlm_score.numpy())