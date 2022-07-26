import easydict

import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

def cut_input(args, tokens):
    if 'roberta' in args.backbone:
        attention_mask = (tokens != 1).float()
    else:
        attention_mask = (tokens > 0).float()
    max_len = int(torch.max(attention_mask.sum(dim=1)))
    return tokens[:, :max_len], attention_mask[:, :max_len], max_len

def get_prediction(model, loader, aug_src):
    all_preds = []

    for i, (tokens, labels, indices) in enumerate(loader):
        batch_size = tokens.size(0)
        if aug_src is not None:
            tokens = aug_src[indices, :]
        tokens, _, _ = cut_input(args, tokens)

        tokens = tokens.cuda()
        labels = labels.cuda()
        labels = labels.squeeze(1)  # (B)

        with torch.no_grad():
            logit = model(tokens, inputs_embed=None)

        preds = logit.softmax(dim=1)[torch.arange(batch_size), labels]

        all_preds.append(preds.cpu())

    return torch.cat(all_preds, dim=0)

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

def lm_recon2(args, tokens, out_ssl):
    '''
    Language modeling reconstruction
    '''
    batch_size = tokens.size(0)

    if 'roberta' in args.backbone:
        attention_mask = (tokens != 1).float().cpu()
    else:
        attention_mask = (tokens > 0).float().cpu()

    num_tokens = attention_mask.sum(dim=1).long()
    attention_mask[:, 0] = 0
    attention_mask[torch.arange(batch_size), num_tokens - 1] = 0

    mask = torch.ones(tokens.size()) * attention_mask.cpu()  # B x L
    num_mask = (num_tokens - 2).cuda() # except start, end tokens
    mask_idx = mask.nonzero()

    labels_ssl = -1 * torch.ones(tokens.size()).to(device).long()  # Sampled : 1, not sampled : -1
    labels_ssl[mask_idx[:, 0], mask_idx[:, 1]] = tokens[mask_idx[:, 0], mask_idx[:, 1]]

    out_ssl = out_ssl.permute(0, 2, 1)

    # lm_acc
    _, pred_ssl = out_ssl.max(dim=1)
    mask_ssl = (labels_ssl != -1).float()
    corrects = (pred_ssl == labels_ssl).float() * mask_ssl
    acc_ssl = corrects.sum() / num_mask.sum()

    # self-supervision loss
    loss_ssl = F.cross_entropy(out_ssl, labels_ssl, ignore_index=-1, reduction='none')  # ignore non-masks (-1)
    loss_ssl = loss_ssl.sum(1) / num_mask

    return loss_ssl.mean(), acc_ssl, num_mask.sum()

def get_mlm_scores_jh(args, model, loader):
    mask_token_idx = 50264

    all_loss = []
    for i, (tokens, labels, indices) in enumerate(loader):
        if (i+1) % 10000 == 0:
            print("{} samples are processed".format(i+1))
        batch_size = tokens.size(0)
        tokens, attention_mask, max_len = cut_input(args, tokens)
        num_tokens = attention_mask.sum(dim=1).long() - 2
        tokens = tokens.cuda()

        mask_loss = torch.zeros(batch_size)
        for j in range(1, max_len - 1):
            masked_tokens = tokens.clone()
            masked_tokens[:, j] = mask_token_idx

            mask = torch.zeros(masked_tokens.size())
            mask[:, j] = 1

            mask_idx = mask.nonzero()
            labels_ssl = -1 * torch.ones(tokens.size()).cuda().long()  # Sampled : 1, not sampled : -1
            labels_ssl[mask_idx[:, 0], mask_idx[:, 1]] = tokens[mask_idx[:, 0], mask_idx[:, 1]]

            with torch.no_grad():
                out_ssl = model(masked_tokens, lm=True)
                out_ssl = out_ssl.permute(0, 2, 1)

            loss_ssl = F.cross_entropy(out_ssl, labels_ssl, ignore_index=-1, reduction='none')[:, j].cpu()
            loss_ssl[num_tokens < j] = 0
            mask_loss += loss_ssl
        mask_loss /= num_tokens
        all_loss.append(mask_loss)

    return -1 * torch.cat(all_loss)


###########################################################################

def get_sentence_embedding(args, model, loader, aug_src=None, head=None):
    all_sentence_embedding = []
    all_labels_embedding = []
    for i, (tokens, labels, indices) in enumerate(loader):
        attention_mask = (tokens > 0).float()

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
                model_output = model(tokens, get_penul=True, sent=True)
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
    n_components = int(np.sum(y < 0.99))

    pca = PCA(n_components=n_components)
    pca.fit(embeddings)

    log_likelihood = pca.score_samples(embeddings)

    print("number of components: {}".format(n_components))

    return pca.transform(embeddings), log_likelihood

def compute_pairwise_distance(data_x, data_y):
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
def compute_nearest_neighbour_distances(input_features_infer, input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features_infer, input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii

def compute_nearest_neighbour_distances_cls(input_features_infer, labels_infer, input_features, labels, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        labels: numpy.ndarray([N], dtype=np.int64)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours "within each class".
    """
    n_class = np.max(labels) + 1
    radii_all = np.zeros((len(input_features_infer)))

    for i in range(n_class):
        i_cls_idx_infer = (labels_infer == i)
        input_features_i_cls_infer = input_features_infer[i_cls_idx_infer]

        i_cls_idx = (labels == i)
        input_features_i_cls = input_features[i_cls_idx]

        distances_i_cls = compute_pairwise_distance(input_features_i_cls_infer, input_features_i_cls)
        radii_i_cls = get_kth_value(distances_i_cls, k=nearest_k + 1, axis=-1)

        radii_all[i_cls_idx] = radii_i_cls

    return radii_all

# From https://github.com/clovaai/generative-evaluation-prdc
def compute_nearest_neighbour_distances_relative(input_features_infer, labels_infer, input_features, labels, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        labels: numpy.ndarray([N], dtype=np.int64)
        nearest_k: int
    Returns:
        "Relative" distance to kth nearest neighbours "between different classes".
        (E.g., Closer to other cls & Far from original cls => Lower score)
    """
    distances = compute_pairwise_distance(input_features_infer, input_features)
    n_class = np.max(labels) + 1
    n_samples = len(distances)

    distances_cls = np.zeros((n_samples, n_class))
    for i in range(n_class):
        i_cls_idx = (labels == i)

        distances_to_i_cls = distances[:, i_cls_idx]
        radii_to_i_cls = get_kth_value(distances_to_i_cls, k=nearest_k + 1, axis=-1)

        distances_cls[:, i] = radii_to_i_cls

    radii_cls = distances_cls[np.arange(n_samples), labels_infer]
    radii_relative = (distances_cls.sum(axis=-1) - radii_cls) / (n_class - 1) - radii_cls

    return radii_relative

def get_density_score(args, sentence_embeddings_infer, labels_infer, sentence_embeddings=None, labels=None):
    density_measure = args.density_measure
    if sentence_embeddings is None:
        sentence_embeddings = sentence_embeddings_infer
        labels = labels_infer

    if density_measure == 'ppca':
        scores = PPCA(sentence_embeddings)
    elif density_measure == 'gaussian':
        scores = GaussianModel(sentence_embeddings)
    elif density_measure == 'nn_dist':
        # make negative so that larger values are better
        scores = -compute_nearest_neighbour_distances(sentence_embeddings_infer,
                                                      sentence_embeddings, nearest_k=5)
    elif density_measure == 'nn_dist_cls':
        scores = -compute_nearest_neighbour_distances_cls(sentence_embeddings_infer, labels_infer,
                                                          sentence_embeddings, labels, nearest_k=5)
    elif density_measure == 'nn_relative_dist':
        scores = compute_nearest_neighbour_distances_relative(sentence_embeddings_infer, labels_infer,
                                                              sentence_embeddings, labels, nearest_k=1)

    return scores
