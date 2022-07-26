
import easydict
import imageio as imageio

import sklearn
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from models import load_backbone, Classifier
from data import get_base_dataset
from utils import load_augment
from training.common import transform_input_masked, cut_input
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
import seaborn as sns


def get_prediction(model, loader, aug_src):
    all_preds = []

    for i, (tokens, labels, indices) in enumerate(loader):
        batch_size = tokens.size(0)
        if aug_src is not None:
            tokens = aug_src[indices, :]
        tokens, _ = cut_input(args, tokens)

        tokens = tokens.cuda()
        labels = labels.cuda()
        labels = labels.squeeze(1)  # (B)

        with torch.no_grad():
            logit = model(tokens, inputs_embed=None)

        preds = logit.softmax(dim=1)[torch.arange(batch_size), labels]

        all_preds.append(preds.cpu())

    return torch.cat(all_preds, dim=0)


# %%

def get_preds_all(args, loc_ckpt, dataset, backbone, total_epoch=10, aug_src=None):
    model = Classifier(args.backbone, backbone, dataset.n_classes, args.train_type).cuda()
    train_loader = DataLoader(dataset.train_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size)

    all_epoch_preds = torch.zeros(len(dataset.train_dataset), total_epoch)

    for epoch in range(1, total_epoch + 1):
        print("Epoch: {}".format(epoch))
        # Load the saved checkpoint of each epoch
        loc_ckpt_epoch = loc_ckpt + '_epoch' + str(epoch) + '.model'
        model.load_state_dict(torch.load(loc_ckpt_epoch))

        train_preds = get_prediction(model, train_loader, aug_src)
        all_epoch_preds[:, epoch - 1] = train_preds

    print(all_epoch_preds.size())


    # Same naming with the original paper

    # for i in range(2,total_epoch + 1):
    #     temp = all_epoch_preds[:, :i]
    #     mu = temp.mean(dim=1, keepdim=True)
    #     sigma = ((temp - mu) ** 2).mean(dim=1)
    #     sigma = torch.sqrt(sigma)
    #
    #     vis_x = sigma
    #     vis_y = mu
    #
    #     plt.rcParams["figure.figsize"] = (10,8)
    #     plt.scatter(vis_x, vis_y, marker='.')
    #
    #     # Title
    #     title_font = {
    #         'fontsize': 18,
    #         'fontweight': 'bold'
    #     }
    #     plt.xlabel("Variability", fontdict=title_font, labelpad=20)
    #     plt.ylabel("Confidence", fontdict=title_font, labelpad=20)
    #     plt.title('{}(0.1) Epoch {}'.format(args.dataset, i), fontdict=title_font, pad=20)
    #     plt.xlim(0, 0.5)
    #     plt.savefig('{}_{}.png'.format(args.dataset, i))
    #     plt.show()


    mu = all_epoch_preds.mean(dim=1, keepdim=True)
    sigma = ((all_epoch_preds - mu) ** 2).mean(dim=1)
    sigma = torch.sqrt(sigma)

    return mu.squeeze(1), sigma


def _ids_to_masked(token_ids, tokenizer):
    mask_indices = np.array([[mask_pos] for mask_pos in range(len(token_ids))])

    input_ids = token_ids.clone()
    special_tokens_mask = np.array(tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True))
    mask_indices = mask_indices[special_tokens_mask==0,:]

    token_ids_masked_list = []

    mask_token_id = tokenizer._convert_token_to_id(tokenizer.mask_token)
    for mask_set in mask_indices:
        token_ids_masked = token_ids.clone()
        token_ids_masked[mask_set] = mask_token_id
        token_ids_masked_list.append((token_ids_masked, mask_set))

    return token_ids_masked_list

def masking_dataset(dataset, tokenizer, aug_src=None):
    masked_dataset = []
    # NOTE: sometimes idx != sent_idx
    for idx, (token_ids, _, sent_idx) in enumerate(dataset):
        if aug_src is not None:
            token_ids = aug_src[sent_idx, :]
        ids_masked = _ids_to_masked(token_ids, tokenizer)
        masked_dataset += [(
            idx,
            sent_idx,
            ids,    # masked token_ids
            mask_set,
            token_ids[mask_set] # label for mask_set location
        )
            for ids, mask_set in ids_masked]
    return masked_dataset

def get_alps_loss(args, model, tokenizer, loader, aug_src):
    """Obtain masked language modeling loss from [model] for tokens in [inputs].
        Should return batch_size X seq_length tensor."""
    all_alps_loss = []
    pad_token_id = tokenizer._convert_token_to_id(tokenizer.pad_token)
    for i, (tokens, labels, indices) in enumerate(loader):
        if aug_src is not None:
            tokens = aug_src[indices, :]
        attention_mask = (tokens != pad_token_id).float()
        tokens, labels = transform_input_masked(args, tokens, tokenizer)

        tokens = tokens.cuda()
        attention_mask = attention_mask.cuda()
        labels = labels.cuda()
        labels = labels.squeeze(1)  # (B)

        inputs = {}
        inputs["input_ids"] = tokens
        inputs["attention_mask"] = attention_mask
        inputs["labels"] = labels

        with torch.no_grad():
            logits = model(**inputs)[0]
            batch_size, seq_length, vocab_size = logits.size()

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss_batched = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
        loss = loss_batched.view(batch_size, seq_length)

        all_alps_loss.append(loss.cpu())

    return torch.cat(all_alps_loss, dim=0)

def get_alps_scores(args, loc_ckpt, tokenizer, dataset, backbone, total_epoch=10, aug_src=None):
    model = Classifier(args.backbone, backbone, dataset.n_classes, args.train_type).cuda()
    train_loader = DataLoader(dataset.train_dataset, shuffle=False, drop_last=False,
                              batch_size=args.batch_size)
    all_epoch_alps_loss = torch.zeros(len(dataset.train_dataset), len(dataset.train_dataset[0][0]), total_epoch)
    for epoch in range(1, total_epoch + 1):
        print("Epoch: {}".format(epoch))
        # Load the saved checkpoint of each epoch
        loc_ckpt_epoch = loc_ckpt + '_epoch' + str(epoch) + '.model'
        model.load_state_dict(torch.load(loc_ckpt_epoch))

        train_mlm_loss = get_alps_loss(args, model.backbone, tokenizer, train_loader, aug_src)
        all_epoch_alps_loss[:, :, epoch - 1] = train_mlm_loss
    return all_epoch_alps_loss

def get_mlm_loss(args, model, loader, aug_src, scores_per_token):
    """Obtain masked language modeling loss from [model] for tokens in [inputs].
        Should return batch_size X seq_length tensor."""
    #TODO: still remains multiGPU issues, it need to specify which gpu is running on dataset.

    for i, (idx, sent_idx, tokens, masked_positions, token_masked_ids) in enumerate(loader):
        # tokens is masked
        if aug_src is not None:
            tokens = aug_src[sent_idx, :]

        pad_token_id = tokenizer._convert_token_to_id(tokenizer.pad_token)
        attention_mask = (tokens != pad_token_id).float()

        batch_size = idx.shape[0]

        tokens = tokens.cuda()
        attention_mask = attention_mask.cuda()
        masked_positions = torch.tensor(masked_positions).reshape(-1, 1)
        token_masked_ids = torch.tensor(token_masked_ids).reshape(-1)

        masked_positions.cuda()
        token_masked_ids = token_masked_ids.cuda()

        inputs = {}
        inputs["input_ids"] = tokens
        inputs["attention_mask"] = attention_mask

        with torch.no_grad():
            logits = model(**inputs)[0]
            out = logits[list(range(batch_size)), masked_positions.reshape(-1), :] # batch_size * vocab_size
        out = out.log_softmax(dim=-1)   # batch_size * vocab_size
        out = out[list(range(batch_size)), token_masked_ids]  # batch_size * 1

        idx = idx.cpu().tolist()
        masked_positions = masked_positions.squeeze().cpu().tolist()

        scores_per_token[idx, masked_positions] = out.cpu().tolist()
        if i != 0 and i % 100 == 0:
            print(i)

    return scores_per_token

def get_mlm_scores(args, model, loader, dataset, aug_src=None):

    scores_per_token = np.array([[0.0] * (len(dataset.train_dataset[0][0])) for i in range(len(dataset.train_dataset))])
    scores_per_token = get_mlm_loss(args, model.backbone, loader, aug_src, scores_per_token)

    return scores_per_token
# %%


def get_sentence_embedding(args, model, loader, aug_src=None, head=None):
    all_sentence_embedding = []
    all_labels_embedding = []
    for i, (tokens, labels, indices) in enumerate(loader):
        if aug_src is not None:
            tokens = aug_src[indices, :]
        if 'bert' in args.backbone:
            pad_token_id = 0
        else:
            pad_token_id = tokenizer._convert_token_to_id(tokenizer.pad_token)
        attention_mask = (tokens != pad_token_id).float()

        tokens = tokens.cuda()
        attention_mask = attention_mask.cuda()

        # Compute token embeddings
        with torch.no_grad():

            if head == 'LM':
                model_output = model(tokens, return_dict=True)
                # Perform pooling. In this case, mean pooling
                if args.backbone == "sentence_bert":
                    sentence_embeddings = mean_pooling(model_output[0], attention_mask)
                else:
                    sentence_embeddings = mean_pooling(model_output[2], attention_mask)
            elif head == 'SC':
                model_output = model(tokens, get_penul=True)
                sentence_embeddings = model_output[1]
            else:
                print("please enter head 'LM' or 'SC'")

        all_sentence_embedding.append(sentence_embeddings.cpu())
        all_labels_embedding.extend(labels.reshape(-1).cpu().tolist())
    return torch.cat(all_sentence_embedding, dim=0), all_labels_embedding

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    #token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
# From https://github.com/clovaai/generative-evaluation-prdc

def GaussianModel(embeddings):
    gmm = GaussianMixture(n_components=1, reg_covar=1e-05)
    gmm.fit(embeddings)

    log_likelihood = gmm.score_samples(embeddings)
    return log_likelihood


def PPCA(embeddings):
    # calculate number of componenets based on 95% variance retention
    n_components = min(embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    var_ratio = pca.explained_variance_ratio_
    y = np.cumsum(var_ratio)
    n_components = int(np.sum(y < 0.95))

    pca = PCA(n_components=n_components)
    pca.fit(embeddings)

    log_likelihood = pca.score_samples(embeddings)
    return log_likelihood

def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists

def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


# From https://github.com/clovaai/generative-evaluation-prdc
def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii

def get_density_score(args,sentence_embeddings):
    density_measure = args.density_measure
    if density_measure == 'ppca':
        scores = PPCA(sentence_embeddings)
    elif density_measure == 'gaussian':
        scores = GaussianModel(sentence_embeddings)
    elif density_measure == 'nn_dist':
        # make negative so that larger values are better
        scores = -compute_nearest_neighbour_distances(sentence_embeddings,
                                                      nearest_k=5)
    return scores

args = easydict.EasyDict({"batch_size": 8,
                          "backbone": 'roberta',
                          "dataset": 'sst2',
                          "train_type": '0718_base',
                          "aug_type": 'backtrans',
                          "seed": 0,
                          #"pre_ckpt": '/nlp/users/yekyung.kim/git/WhatsUp/logs/sst2_0718_base_S1234/sst2_roberta-base_0718_base_epoch1.model',
                          "mlm_probability":0.15,
                          #"score_type": 'confidence',
                          #"topK": True,
                          #"sub_ratio": 0.33,
                          "density_measure":'nn_dist', #ppca, gaussian, nn_dist
                          "data_ratio":0.1,
                          })
title_font = {
    'fontsize': 18,
    'fontweight': 'bold'
}

#
# backbone, tokenizer = load_backbone(args.backbone)
# dataset, train_loader, val_loader, test_loader = get_base_dataset(args.dataset, tokenizer, batch_size=args.batch_size,\
#                                                                   data_ratio=args.data_ratio, seed=args.seed, shuffle=False)
# # aug_src = load_augment(args, dataset, tokenizer)[:, 0, :]
# # loc_ckpt = '/nlp/users/yekyung.kim/git/WhatsUp/logs/sst2_R0.1_0802_base_S3456/sst2_roberta-base_R0.100_0802_base'
# loc_ckpt = '/nlp/users/yekyung.kim/git/WhatsUp/logs/mnli_R0.1_0802_base_S3456/mnli_roberta-base_R0.100_0802_base'
#
# mu, sigma = get_preds_all(args, loc_ckpt, dataset, backbone, total_epoch=5)


############# calculating MLM loss score ######################

# model = Classifier(args.backbone, backbone, dataset.n_classes, args.train_type).cuda()
# masked_dataset = masking_dataset(dataset.train_dataset, tokenizer)#, aug_src)
# masked_train_loader = DataLoader(masked_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size)
#
# mlm_losses = get_mlm_scores(args, model, masked_train_loader, dataset)
# seq_len = (mlm_losses != 0).sum(1)
# sentence_mlm_scores = mlm_losses.sum(1) / seq_len #batch_size
# sentence_mlm_scores = sentence_mlm_scores.reshape(-1,1)
#
# np.save('./scores/{}_{}_{}_mlm.npy'.format(args.dataset,  args.data_ratio, args.backbone), sentence_mlm_scores)
# # data2 = np.load('./scores/{}_{}_mlm.npy'.format(args.dataset, args.data_ratio))
# print(mlm_losses.shape) #batch_size * max_seq_length

############## calculating density score ######################
# target_models = ['roberta', 'sentence_bert']
# for t_model in target_models:
#     args.backbone = t_model
#     backbone, tokenizer = load_backbone(args.backbone)
#     dataset, train_loader, val_loader, test_loader = get_base_dataset(args.dataset, tokenizer, batch_size=args.batch_size, \
#                                                                       data_ratio=args.data_ratio, seed=args.seed, shuffle=False)
#     train_loader = DataLoader(dataset.train_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size)
#
#     #TODO: augmentation dataset should be calculated with tokenizer of sentence_bert
#     model = backbone.cuda()
#     sentence_embeddings, all_labels = get_sentence_embedding(args, model, train_loader, head='LM')
#
#     ## plot sentence embedding into 2d space
#     # sentence_to_2d = TSNE(n_components=2, random_state=0).fit_transform(sentence_embeddings)
#     # plt.rcParams["figure.figsize"] = (10,8)
#     # # for x,y,label in zip(sentence_to_2d[:, 0], sentence_to_2d[:, 1], all_labels):
#     # #     plt.scatter(x,y, marker='.', label=label)
    #     # colors = ['red', 'green', 'blue', 'purple']
#     # plt.scatter(sentence_to_2d[:, 0], sentence_to_2d[:, 1], c=all_labels, cmap=matplotlib.colors.ListedColormap(colors),marker='.')
#     # plt.legend(colors)
#     # plt.title('T-SNE of {} dataset({})-pretrained_{}'.format(args.dataset, args.data_ratio, args.backbone), fontdict=title_font, pad=20)
#     # plt.savefig('tsne_{}_{}_{}.png'.format(args.dataset, args.data_ratio, args.backbone))
#     # plt.show()
#     # plt.clf()
#
#     for method in ["nn_dist"]:  #["ppca", "gaussian"]:
#         args.density_measure = method
#         print("start calculating density by {}".format(method))
#         density_scores = get_density_score(args, sentence_embeddings)
#         np.save('./scores/{}_{}_{}_density_{}.npy'.format(args.dataset, args.data_ratio, args.backbone, args.density_measure), density_scores)
#     # data2 = np.load('./scores/{}_{}_density_{}.npy'.format(args.dataset, args.data_ratio, args.density_measure))
#     print(density_scores.shape)

#### calulating density dynamics on fine-tuned model
# args.backbone = 'roberta'
# total_epoch = 10
# loc_ckpt = '/nlp/users/yekyung.kim/git/WhatsUp/logs/{}_R0.1_0802_base_S3456/{}_roberta-base_R0.100_0802_base'.format(args.dataset, args.dataset)
#
# backbone, tokenizer = load_backbone(args.backbone)
# dataset, train_loader, val_loader, test_loader = get_base_dataset(args.dataset, tokenizer, batch_size=args.batch_size, \
#                                                                   data_ratio=args.data_ratio, seed=args.seed, shuffle=False)
# train_loader = DataLoader(dataset.train_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size)
# model = Classifier(args.backbone, backbone, dataset.n_classes, args.train_type).cuda()
#
# all_epoch_density = np.zeros((len(dataset.train_dataset), total_epoch))
# for epoch in range(1, total_epoch + 1):
#     print("Epoch: {}".format(epoch))
#     loc_ckpt_epoch = loc_ckpt + '_epoch' + str(epoch) + '.model'
#     model.load_state_dict(torch.load(loc_ckpt_epoch))
#     sentence_embeddings, all_labels = get_sentence_embedding(args, model, train_loader, head='SC')
#
#     if epoch == total_epoch:
#         sentence_to_2d = TSNE(n_components=2, random_state=0).fit_transform(sentence_embeddings)
#         plt.rcParams["figure.figsize"] = (10,8)
#         # for x,y,label in zip(sentence_to_2d[:, 0], sentence_to_2d[:, 1], all_labels):
#         #     plt.scatter(x,y, marker='.', label=label)
#         colors = ['red', 'green', 'blue', 'purple']
#         plt.scatter(sentence_to_2d[:, 0], sentence_to_2d[:, 1], c=all_labels, cmap=matplotlib.colors.ListedColormap(colors),marker='.')
#         plt.legend(colors)
#         plt.title('T-SNE of {} dataset({})-{},{}epoch'.format(args.dataset, args.data_ratio, args.backbone, total_epoch), fontdict=title_font, pad=20)
#         plt.savefig('tsne_{}_{}_{}_{}.png'.format(args.dataset, args.data_ratio, args.backbone, total_epoch))
#         plt.show()
#         plt.clf()
#
#     density_scores = get_density_score(args, sentence_embeddings)
#     all_epoch_density[:, epoch - 1] = density_scores.tolist()
#
# np.save('./scores/{}_{}_{}_density_dynamics_{}.npy'.format(args.dataset, args.data_ratio, args.backbone, args.density_measure), all_epoch_density)
# print(density_scores.shape)

######## plot all scores we have ########
# data = 'mnli'
# ratio = 0.1
# base_model = 'roberta'
#
# temp =  np.load('./scores/{}_{}_{}_{}.npy'.format(data, ratio, base_model, 'badge'))
#
# measurements = ['avg_conf', 'variability', 'forget_number', 'aum', 'density_nn_dist', 'mlm', 'conf', 'Ent',
#                 'knn_density_target', 'ens_ent', 'ens_bald', 'ens_varR']#, 'density_dynamics_nn_dist']
# dict = {}
# for measurement in measurements:
#     print(measurement)
#     dict[measurement] = np.load('./scores/{}_{}_{}_{}.npy'.format(data, ratio, base_model, measurement))
#     print(dict[measurement].shape)
#
# for idx, x_measurement in enumerate(measurements):
#     x_name = x_measurement
#     for y_measurement in measurements[idx+1:]:
#
#         x_name = x_measurement
#         y_name = y_measurement
#
#         print("plot x: {}, y: {}".format(x_name, y_name))
#
#         vis_x = dict[x_name]
#         vis_y = dict[y_name]
#
#         plt.rcParams["figure.figsize"] = (10,8)
#         plt.scatter(vis_x, vis_y, marker='.')
#
#         # Title
#         title_font = {
#             'fontsize': 18,
#             'fontweight': 'bold'
#         }
#         plt.xlabel(x_name, fontdict=title_font, labelpad=20)
#         plt.ylabel(y_name, fontdict=title_font, labelpad=20)
#         plt.title('{} dataset(ratio: {})'.format(data, ratio), fontdict=title_font, pad=20)
#         plt.show()
#

########## plotting trajectory with gif ##########
# args.dataset = "sst2"
# if args.dataset == "sst2":
#     epoch = 10
# elif args.dataset =="mnli":
#     epoch = 5
# with imageio.get_writer('{}_trajectory.gif'.format(args.dataset), mode='I') as writer:
#     for epoch in range(2, epoch+1):
#         filename = '{}_{}.png'.format(args.dataset, epoch)
#         image = imageio.imread(filename)
#         writer.append_data(image)

########### plotting relationship between method of density meaurement
# data = "mnli"
# ratio = "0.1"
# base_model = "roberta"
# densities = {}
# methods = ["ppca","gaussian","nn_dist"]
# for method in methods:
#     densities[method] = np.load('./scores/{}_{}_{}_density_{}.npy'.format(data, ratio, base_model, method))
#
# x = "ppca"
# y = "gaussian"
# plt.rcParams["figure.figsize"] = (10,8)
# plt.scatter(densities[x], densities[y], marker='.')
#
# plt.xlabel(x, fontdict=title_font, labelpad=20)
# plt.ylabel(y, fontdict=title_font, labelpad=20)
# plt.title('{} dataset(ratio: {})'.format(data, ratio), fontdict=title_font, pad=20)
# plt.show()

from mpl_toolkits import mplot3d


# %matplotlib notebook
data = "mnli"
ratio = 0.1
base_model = "roberta"
measurements = ['avg_conf', 'variability', 'forget_number', 'aum', 'density_nn_dist', 'mlm', 'conf', 'Ent',
                'knn_density_target', 'ens_ent', 'ens_bald', 'ens_varR', 'density_gaussian', 'density_ppca']#, 'density_dynamics_nn_dist']
dict = {}
for measurement in measurements:
    print(measurement)
    dict[measurement] = np.load('./scores/{}_{}_{}_{}.npy'.format(data, ratio, base_model, measurement))
    print(dict[measurement].shape)
x = "avg_conf"
y = "variability"
z = "density_gaussian"
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(dict[x], dict[y], dict[z])#, c=zdata, cmap='Greens')

ax.set_xlabel(x)
ax.set_ylabel(y)
ax.set_zlabel(z)
plt.title('{} dataset(ratio: {})'.format(data, ratio), fontdict=title_font, pad=20)
plt.show()
