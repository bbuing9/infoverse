## test 

import os
import easydict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import scipy
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from models import load_backbone, Classifier

from data import get_base_dataset
from utils import Logger, set_seed, set_model_path, save_model, load_augment, add_mislabel_dataset
from training.common import cut_input, get_embed, data_aug

from scores_src import get_features, merge_multiple_models, surprisal_embed, surprisal_embed_wino
from scores_src import avg_conf_variab, avg_forgetting, avg_aum
from scores_src import get_density_score, get_mlm_scores, masking_dataset, get_mlm_scores_jh, get_sentence_embedding, \
    PPCA, compute_nearest_neighbour_distances_cls
from scores_src import confidence, entropy, badge_grads_norm, badge_grads
from scores_src.ensembles import mc_dropout_models, el2n_score, ens_max_ent, ens_bald, ens_varR
from scores_src.dpp import gaussian_kernel, dpp_greedy

def dpp_samping(n_query, measurements, labels, scores_type='density', reduce=False):
    n_sample = len(measurements)
    eps = 5e-2

    # Dimension reduction for removing redundant features
    if reduce:
        info_measures, _ = PPCA(measurements)
    else:
        info_measures = np.array(measurements)

    # Define similarity kernel phi(x_1, x_2)
    similarity = gaussian_kernel(info_measures / np.linalg.norm(info_measures, axis=-1).reshape(-1, 1))

    # Define score function q(x)
    if scores_type == 'density':
        scores_bef = -1 * compute_nearest_neighbour_distances_cls(info_measures, labels, info_measures, labels, nearest_k=5)
        scores = -1 / (1e-8 + scores_bef)
    elif scores_type == 'hard':
        scores = np.sqrt(-1 * np.log(measurements[:, 0] + 1e-8))
    elif scores_type == 'ambig':
        scores = np.sqrt(measurements[:, 1])
    else:
        scores = np.ones(n_sample)
    scores = (scores - scores.min()) / scores.max()

    dpp_kernel = scores.reshape((n_sample, 1)) * similarity * scores.reshape((1, n_sample))
    selected_idx = dpp_greedy(dpp_kernel + eps * np.eye(n_sample), n_query, save=True)

    return selected_idx

args = easydict.EasyDict({"batch_size": 16,
                          "backbone": 'roberta_large',
                          "dataset": 'qnli',
                          "ood_dataset": 'trec',
                          "train_type": '1230_base_large',
                          "aug_type": 'none',
                          "seed": 1234,
                          "name": '1230_base_large',
                          "pre_ckpt": './logs/cola_R1.0_1230_base_large_S1234/cola_roberta-large_1230_base_large_epoch5.model',
                          "score_type": 'confidence',
                          "topK": True,
                          "data_ratio": 1.0,
                          "n_classes": 2,
                          "noisy_label_criteria": 'avg_conf',
                          "noisy_label_ratio": 0.0,
                        })
#backbone, tokenizer = load_backbone(args.backbone)
#dataset, train_loader, val_loader, test_loader = get_base_dataset(args.dataset, tokenizer, args.batch_size, args.data_ratio, args.seed, shuffle=False)
#labels_t = dataset.train_dataset[:][1][:, 0].numpy()
labels_t = np.load('0105_qnli_labels_t.npy')

# Load measurements
eps = 1e-8
#measures = np.load('0105_qnli_large_1.0_all_measurements_true.npy')
#print(measures.shape)
#measures_norm = (measures - measures.mean(axis=0)) / (measures.std(axis=0) + eps)
measures_norm = np.load('0105_qnli_penuls.npy')
dpp_lists = dpp_samping(int(0.83 *len(measures_norm)), measures_norm.astype(np.float32), labels_t, scores_type='density')

print(len(dpp_lists))
#np.save('0105_qnli_large_info_true_dens_dpp.npy', dpp_lists)
np.save('0105_qnli_large_penuls_dens_dpp.npy', dpp_lists)

