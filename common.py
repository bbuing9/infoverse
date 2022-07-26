import argparse

DATA_PATH = './dataset'
CKPT_PATH = './checkpoint'

def parse_args(mode):
    assert mode in ['train', 'eval']

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help='dataset (news|review|imdb|etc.)',
                        required=True, type=str)
    parser.add_argument("--backbone", help='backbone network',
                        choices=['bert', 'roberta', 'roberta_large', 'roberta_mc', 'roberta_mc_large', 'albert'],
                        default='bert', type=str)
    parser.add_argument("--data_ratio", help='data ratio',
                        default=1.0, type=float)
    parser.add_argument("--seed", help='random seed',
                        default=0, type=int)

    parser = _parse_args_train(parser)

    return parser.parse_args()


def _parse_args_train(parser):
    # ========== Training ========== #
    parser.add_argument("--train_type", help='train type (base|aug|mixup|lad2)',
                        default='base', type=str)
    parser.add_argument("--epochs", help='training epochs',
                        default=15, type=int)
    parser.add_argument("--batch_size", help='training bacth size',
                        default=8, type=int)
    parser.add_argument("--model_lr", help='learning rate for model update',
                        default=1e-5, type=float)
    parser.add_argument("--save_ckpt", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--linear", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--grad_accumulation", help='inverval for model update',
                        default=1, type=int)

    parser.add_argument("--pre_ckpt", help='path for the pre-trained model',
                        default=None, type=str)
    parser.add_argument("--selected", help='path to selected subset of augmented samples',
                        default=None, type=str)
    parser.add_argument("--selected_idx", help='path to selected subset of augmented samples',
                        default=None, type=str)

    # ========== Data Valuation via Reinforcement Learning ========== #
    parser.add_argument("--original", help='original DVRL implementation',
                        action='store_true')
    parser.add_argument("--perf_metric", help='candidates: auc|accuracy|log_loss ',
                        default='log_loss', type=str)

    parser.add_argument("--pre_load", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--init_lin", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--normalize", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--logit", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--pre", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--logit2", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--no_diff", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--epoch_wise", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--all_val", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--no_regular", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--measurement", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--measurement2", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--base", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--move_base", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--save_valuation", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--linear_value", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--new_linear", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--continuous", help='save the best model checkpoint',
                        action='store_true')

    parser.add_argument("--pre_ckpt_val", help='path for the pre-trained model of validation data',
                        default=None, type=str)
    parser.add_argument("--enlarge", help='enlarge batch-size for efficient learning',
                        default=16, type=int)
    parser.add_argument("--pre_epochs", help='training epochs',
                        default=15, type=int)
    parser.add_argument("--window_size", help='window size for moving average',
                        default=0, type=int)
    parser.add_argument("--n_batch_reward", help='training epochs',
                        default=0, type=int)
    parser.add_argument("--estim_lr", help='learning rate for policy update',
                        default=1e-5, type=float)
    parser.add_argument("--lambda_reg", help='hyper-parameter for regularization',
                        default=1000, type=float)
    parser.add_argument("--inner_iterations", help='number of training iterations for updating model',
                        default=100, type=int)
    parser.add_argument("--outer_iterations", help='number of training iterations for updating model',
                        default=100, type=int)


    # ========== Data Pruning ========== #
    parser.add_argument("--num_sub", help='number of sub samples',
                        default=1.0, type=float)
    parser.add_argument("--descending", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--sub_easy", help='save the best model checkpoint',
                        action='store_true')

    # ========== Data Augmentations ========== #
    parser.add_argument("--aug_type", help='augmentation type',
                        choices=['backtrans', 'cutoff', 'eda', None],
                        default=None, type=str)

    parser.add_argument("--mixup_alpha", help='hyper-parameter of beta distribution for Mixup augmentation',
                        default=1.0, type=float)
    parser.add_argument("--cutoff", help='length of cutoff tokens',
                        default=0.10, type=float)
    parser.add_argument("--eps", help='random noise size for r3f',
                        default=1e-5, type=float)
    parser.add_argument("--step_size", help='step size for adversarial example',
                        default=0.1, type=float)

    # ========== LAD^2 ========== #
    parser.add_argument("--lambda_cls", help='weight for classification loss',
                        default=1.0, type=float)
    parser.add_argument("--lambda_aug", help='weight for classification loss for augmented samples',
                        default=1.0, type=float)
    parser.add_argument("--lambda_kl", help='weight for symmetric KL divergence loss (consistency)',
                        default=0.0, type=float)

    # ========= Mislabeled ========= #
    parser.add_argument("--noisy_oracle", help='remove noisy labeled data for training, i.e., Oracle',
                        action='store_true')
    parser.add_argument("--noisy_label_criteria", help="Whether to add noisy label.",
                        default=None, type=str, choices=['avg_conf', 'random'])
    parser.add_argument("--noisy_label_ratio", help='ratio of noisy label',
                        default=0.0, type=float)
    parser.add_argument("--noisy_label_path", help="path of predefined noise label ",
                        default=None, type=str)

    return parser

