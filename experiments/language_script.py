# coding: utf-8
import argparse
import time
import math
import os
import sys
sys.path.append(os.getcwd()) # Fix Python Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from language_model import RNNModel, Corpus

from tbptt import (
    TBPTT_minibatch_helper,
    )
from tbptt.adaptive_truncation import (
    calculate_gradient_norms,
    calculate_component_gradient_norms,
    adaptive_K_estimate,
    log_estimate_grad_norm,
    )

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Modified from pytorch/examples/word_language_model on Github
parser = argparse.ArgumentParser(description='PyTorch Word Language Model')
# I/O args
parser.add_argument("--experiment_id", type=int, default=-1)
parser.add_argument("--experiment_folder", type=str, default='test')
# Data Args
parser.add_argument('--data_name', type=str, default='./data/wikitext-2',
                    help='location of the data corpus (contains train.txt, valid.txt, test.txt)')
# Model Args
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')

# Training Args
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--scale_lr_K', action='store_true',
                    help='scale learning rate by K')
parser.add_argument('--decay_lr', action='store_true',
                    help='decay learning rate by 1/sqrt(epoch)')
parser.add_argument('--weight_decay', type=float, default=10**-6,
                    help='L2 regularization')
parser.add_argument('--clip_grad', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=35,
                    help='upper epoch limit')
parser.add_argument('--max_train_time', type=float, default=3*3600,
                    help='max training time')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument("--tbptt_style", type=str, default='original_buffer',
        help="One of (tf, buffer, original, original-buffer, double-tf)")
parser.add_argument('--K', type=int, default=25,
                    help='TBPTT sequence length')
parser.add_argument('--adaptive_K', action='store_true',
                    help='use adaptive K')
parser.add_argument('--delta', type=float, default=0.1,
        help='adaptive_K relative bias parameter')
parser.add_argument("--beta_estimate_method", type=str, default=None,
        help="{'max', 'ols', 'quantile'}")
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed')
parser.add_argument('--init_num', type=int, default=0,
                    help='initial parameters')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--linear_scale', action='store_true',
                    help='scale stepsize by K rather than sqrt(K)')
args = parser.parse_args()
print(args)

# Additional Training CONSTs
TP_SEQ_LEN = 5000
MAX_TRAIN_STEPS = 1000
MAX_TIME_PER_STEP = 120
MAX_STEPS_PER_STEP = 40
NUM_K_EST_REPS = 10
TAUS_RANGE = np.arange(150, 351, 50)
MIN_K = 10
MAX_K = 300
CHECK_FREQ = 10
GRAD_CHECK_FREQ = 10
GRAD_CHECK_SEQ_LEN = 400
K_EST_SEQ_LEN = 400


# Set the random seed manually for reproducibility.
if args.seed is None:
    torch.manual_seed(args.experiment_id)
else:
    torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

# Setup Paths for Figures + Output
method_name = '{0}_{1}_{2}_{3}_{4}'.format(
    args.experiment_id,
    args.model,
    args.K if not args.adaptive_K else 'adaptive{0}'.format(args.delta),
    args.tbptt_style,
    args.lr,
    )
path_to_out = os.path.join(args.experiment_folder, 'out', method_name)
if not os.path.isdir(path_to_out):
    os.makedirs(path_to_out)
joblib.dump(pd.DataFrame([vars(args)]), os.path.join(path_to_out, 'options.p'))
path_to_init_param_dict = os.path.join(args.experiment_folder, 'out',
        "init_{0}_state_dict.pth".format(args.init_num))
path_to_check_param_dict = os.path.join(path_to_out,"checkpoint_state_dict.pth")


###############################################################################
# Load data
###############################################################################
print("Loading Data")
corpus = Corpus(args.data_name)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
estK_data = batchify(corpus.train, 64)
valid_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)


###############################################################################
# Build the model
###############################################################################
print("Setting Up Model Module")
ntokens = len(corpus.dictionary)
rnn_module = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

# Copy Initialization between different TBPTT(K) methods
if os.path.isfile(path_to_check_param_dict):
    print("Loading Module Parameters from Checkpoing {0}".format(
        path_to_check_param_dict))
    rnn_module.load_state_dict(torch.load(path_to_check_param_dict))
    if os.path.isfile(os.path.join(path_to_out, 'metrics.p.gz')):
        metric_df = [joblib.load(os.path.join(path_to_out, 'metrics.p.gz'))]
    else:
        metric_df = []
elif os.path.isfile(path_to_init_param_dict):
    print("Loading Module Init Parameters from {0}".format(
        path_to_init_param_dict))
    rnn_module.load_state_dict(torch.load(path_to_init_param_dict))
    metric_df = []
else:
    print("Saving Module Init Parameters to {0}".format(
        path_to_init_param_dict))
    torch.save(rnn_module.state_dict(), path_to_init_param_dict)
    metric_df = []

# Loss + Optimizer
optimizer = torch.optim.SGD(
        rnn_module.parameters(),
        lr = args.lr,
        weight_decay = args.weight_decay,
        )
loss_module = nn.CrossEntropyLoss()

# Minibatch Helper (calls train() + eval())
runner = TBPTT_minibatch_helper(
        rnn_module=rnn_module,
        loss_module=loss_module,
        optimizer=optimizer,
        use_buffer=(args.tbptt_style == 'buffer'),
        original_style=(args.tbptt_style == 'original'),
        original_buffer=(args.tbptt_style == 'original-buffer'),
        double_tf=(args.tbptt_style == 'double-tf'),
        cuda=args.cuda,
        normed=True,
        )

###############################################################################
# Training code
###############################################################################
print("Training Loop")

# Helper Functions
def get_batch(source, start, seq_len):
    seq_len = min(seq_len, source.shape[0] - 1 - start)
    X = source[start:start+seq_len]
    Y = source[start+1:start+1+seq_len]
    return X, Y

def stochastic_subset(source, seq_len):
    """ Subsample seq_len uniformly for each element along dim 1 """
    shifts=np.random.randint(0, source.shape[0]-seq_len-1, size=source.shape[1])
    subset_X = torch.cat([source[shift:shift+seq_len,ii:ii+1]
        for (ii, shift) in enumerate(shifts)], dim=1)
    subset_Y = torch.cat([source[shift+1:shift+1+seq_len,ii:ii+1]
        for (ii, shift) in enumerate(shifts)], dim=1)
    return subset_X, subset_Y


def beta_est_method_to_params(beta_estimate_method):
    """ One of the following
        'max' or 'ols'
        """
    beta_est_kwargs = {
            'max': dict(
                beta_estimate_method='max',
                est_after_log=False,
                ),
            'max-post': dict(
                beta_estimate_method='max',
                est_after_log=True,
                ),
            'ols': dict(
                beta_estimate_method='ols',
                est_after_log=False,
                ),
            'ols-post': dict(
                beta_estimate_method='ols',
                est_after_log=True,
                ),
            'quantile': dict(
                beta_estimate_method='quantile',
                est_after_log=False,
                ),
            }
    if beta_estimate_method not in beta_est_kwargs:
        raise ValueError("Unrecognized beta_estimate_method")
    return beta_est_kwargs[beta_estimate_method]

def adaptive_K_est(source, runner,
        seq_len, burnin,
        tau, deltas=[],
        beta_estimate_methods=None,
        W=None, trun_range=None):
    X, Y = stochastic_subset(source, seq_len=seq_len+burnin)
    grad_norm, cum_grad_norm = calculate_gradient_norms(
            X=X[burnin:],
            Y=Y[burnin:],
            init_state = runner.rnn_module.get_burnin_init(X[:burnin]),
            rnn_module = runner.rnn_module,
            loss_module = runner.loss_module,
            cum_grad_norm_approx = True,
            tqdm=tqdm,
            )
    out = dict()
    if beta_estimate_methods is None:
        beta_estimate_methods = ['ols']

    for beta_estimate_method in beta_estimate_methods:
        beta_kwargs = beta_est_method_to_params(beta_estimate_method)
        K_est = adaptive_K_estimate(grad_norm, cum_grad_norm, tau,
                deltas=deltas, **beta_kwargs)
        for delta, K in zip(deltas, K_est):
            out['{0}_{1}'.format(beta_estimate_method, delta)] = K
    return out

def calc_metric_estimate(source, runner,
        cur_K,
        seq_len, burnin,
        tau, deltas=[],
        beta_estimate_method=None,
        W=None, trun_range=None):
    X, Y = stochastic_subset(source, seq_len=seq_len+burnin)
    grad_norm, cum_grad_norm = calculate_gradient_norms(
            X=X[burnin:],
            Y=Y[burnin:],
            init_state = runner.rnn_module.get_burnin_init(X[:burnin]),
            rnn_module = runner.rnn_module,
            loss_module = runner.loss_module,
            cum_grad_norm_approx = True,
            tqdm=tqdm,
            )
    out = dict()
    if beta_estimate_method is None:
        beta_estimate_methods = ['ols']
    else:
        beta_estimate_methods = [beta_estimate_method]

    for beta_estimate_method in beta_estimate_methods:
        beta_kwargs = beta_est_method_to_params(beta_estimate_method)
        extra_out = adaptive_K_estimate(grad_norm, cum_grad_norm, tau,
                deltas=deltas, extra_out=True, **beta_kwargs)

        # K estimates
        for delta, K in zip(deltas, extra_out['K_est']):
            out['K_{0}'.format(delta)] = K
        out['cur_logdelta'] = extra_out['log_rel_error'][cur_K-1]
    return out

def checkpoint_grad_plots(runner, source, path_to_figures,
        taus = TAUS_RANGE, seq_len=GRAD_CHECK_SEQ_LEN,
        start=0, burnin=50):
    plt.close('all')
    train_X, train_Y = get_batch(
            source=source, start=start, seq_len=seq_len+2*burnin,
            )
    train_init_state = runner.rnn_module.get_burnin_init(train_X[:burnin])

    # Component-wise Grad Norm Plots
    grad_norms, cum_grad_norms, component_names = calculate_component_gradient_norms(
            X=train_X[burnin:],
            Y=train_Y[burnin:],
            init_state = train_init_state,
            rnn_module = runner.rnn_module,
            loss_module = runner.loss_module,
            cum_grad_norm_approx = True,
            tqdm=tqdm,
            )
    # Save Grads
    joblib.dump(dict(
        grad_norms=grad_norms,
        cum_grad_norms=cum_grad_norms,
        component_names=component_names,
        ),
        os.path.join(path_to_figures, 'grads.p'),
        compress=True,
        )
    for grad_norm, cum_grad_norm, key in tqdm(
            zip(grad_norms, cum_grad_norms, component_names),
            desc='grad norm plots',
            total=len(component_names)):
        path_to_grad_norm = os.path.join(path_to_figures, 'grad_{0}'.format(key))
        if not os.path.isdir(path_to_grad_norm):
            os.makedirs(path_to_grad_norm)

        log_grad_norms = np.log(grad_norm + 1e-100)
        log_grad_norms[log_grad_norms < -50] = np.NaN

        est_abs_bias = np.abs(np.cumsum(grad_norm, axis=0)-np.sum(grad_norm, axis=0))
        est_rel_bias = est_abs_bias/(np.cumsum(grad_norm, axis=0)-est_abs_bias)
        est_rel_bias[est_rel_bias < 0] = np.nan

        # Plot Grad Norms + Log Grad Norm Diff
        fig, axes = plt.subplots(2,1)
        axes[0].plot(np.diff(log_grad_norms, axis=0)[:-burnin],
                color='gray', alpha=0.2)
        axes[0].axhline(y=0, linestyle="--", color='black')
        axes[0].plot(np.nanmedian(np.diff(log_grad_norms, axis=0), axis=1)[:-burnin],
                color='C0')
        axes[0].plot(np.nanmean(np.diff(log_grad_norms, axis=0), axis=1)[:-burnin],
                color='C1', linestyle="--", alpha=0.9)
        axes[0].set_ylabel("Log Grad Norm Diff")
        axes[0].set_xlabel("Lag")

        axes[1].plot(grad_norm[:-burnin], color='gray', alpha=0.2)
        axes[1].plot(np.median(grad_norm, axis=1)[:-burnin], color='C0')
        axes[1].plot(np.mean(grad_norm, axis=1)[:-burnin], color='C1',
                linestyle='--', alpha=0.9)
        axes[1].plot(np.diff(cum_grad_norm)[:-burnin], color='C2')
        axes[1].fill_between(x=np.arange(seq_len),
                y1 = np.quantile(grad_norm, 0.05, axis=1)[:-burnin],
                y2 = np.quantile(grad_norm, 0.95, axis=1)[:-burnin],
                color='C0', alpha=0.2)
        axes[1].set_yscale('log')
        axes[1].set_ylabel("Grad Norm")
        axes[1].set_xlabel("Lag")
        fig.savefig(os.path.join(path_to_grad_norm, 'grad_plot_{0}.png'.format(key)))


        # Estimates of Relative Error
        beta_estimate_methods = ['ols']

        beta_trun_abs_bias = {}
        beta_trun_rel_bias = {}
        beta_log_grad_norm = {}
        for beta_estimate_method in beta_estimate_methods:
            plt.close('all')
            beta_kwargs = beta_est_method_to_params(beta_estimate_method)
            extra_outs = [
                    adaptive_K_estimate(grad_norm, cum_grad_norm,
                        tau=tau,
                        deltas=[0.1],
                        trun_range=np.arange(seq_len)+1,
                        extra_out=True,
                        **beta_kwargs
                        )
                    for tau in taus]

            log_est_grad_norm = np.array([log_estimate_grad_norm(
                    grad_norm=grad_norm,
                    logbeta = out_['logbeta'],
                    tau=tau,
                    trun_range=np.arange(seq_len)+1,
                    **beta_kwargs
                )
                for out_, tau in zip(extra_outs, taus)])

            log_est_min = np.min(log_est_grad_norm, axis=0)
            beta_log_grad_norm[beta_estimate_method] = log_est_min

            trun_rel_bias = np.array([np.exp(out_['log_rel_error'])
                for out_ in extra_outs])
            trun_rel_bias_min = np.min(trun_rel_bias, axis=0)
            beta_trun_rel_bias[beta_estimate_method] = trun_rel_bias_min

            trun_abs_bias = np.array([
                np.exp(out_['log_abs_error'])
                for out_ in extra_outs])
            trun_abs_bias_min = np.min(trun_abs_bias, axis=0)
            beta_trun_abs_bias[beta_estimate_method] = trun_abs_bias_min

            # Plot Rel Bias for each beta estimate method
            fig, axes = plt.subplots(4,1)
            est_rel_bias = 1-np.cumsum(grad_norm, axis=0)/np.sum(grad_norm, axis=0)
            est_abs_bias = np.abs(np.cumsum(grad_norm, axis=0)-np.sum(grad_norm, axis=0))

            axes[0].plot(np.diff(log_grad_norms, axis=0)[:-burnin], color='gray', alpha=0.1)
            for tau, log_est_norm in zip(taus, log_est_grad_norm):
                axes[0].plot(np.arange(tau-10+1, len(log_est_norm)),
                        log_est_norm[tau-10+1:] - log_est_norm[tau-10:-1],
                        label='tau={0}'.format(tau), linestyle='-', alpha=0.95)
            axes[0].legend()
            axes[0].axhline(y=0, linestyle="--", color='black')
            axes[0].set_ylabel("Log Grad Norm Diff Est")
            axes[0].set_xlabel("Lag")

            axes[1].plot(np.arange(1,seq_len+1), grad_norm[:-burnin], color='gray', alpha=0.1)
            axes[1].plot(np.arange(1,len(log_est_min)+1), np.exp(log_est_min), label='best', color='k', linewidth=3)
            for tau, log_est_norm in zip(taus, log_est_grad_norm):
                axes[1].plot(np.arange(1, len(log_est_norm)+1),
                        np.exp(log_est_norm),
                        label='tau={0}'.format(tau), linestyle='--', alpha=0.7)
            axes[1].legend()
            axes[1].set_ylabel("Grad Norm Est")
            axes[1].set_xlabel("Lag")
            axes[1].set_yscale('log')

            axes[2].plot(np.arange(1,seq_len+1), est_abs_bias[:-burnin], color='gray', alpha=0.1)
            axes[2].plot(np.arange(1,len(trun_abs_bias_min)+1), trun_abs_bias_min, label='best', color='k', linewidth=3)
            for tau, trun_abs_bias_ in zip(taus, trun_abs_bias):
                axes[2].plot(np.arange(1, len(trun_abs_bias_)+1), trun_abs_bias_,
                        label='tau={0}'.format(tau), linestyle='--', alpha=0.7)
            axes[2].legend()
            axes[2].set_ylabel("Est Abs Bias")
            axes[2].set_xlabel("Truncation Length")
            axes[2].set_yscale('log')

            axes[3].plot(np.arange(1,seq_len+1), est_rel_bias[:-burnin], color='gray', alpha=0.1)
            axes[3].plot(np.arange(1,len(trun_rel_bias_min)+1),
                    trun_rel_bias_min, label='best', color='k', linewidth=3)
            for tau, trun_rel_bias_ in zip(taus, trun_rel_bias):
                axes[3].plot(np.arange(1, len(trun_rel_bias_)+1), trun_rel_bias_,
                        label='tau={0}'.format(tau), linestyle='--', alpha=0.7)
            axes[3].legend()
            axes[3].set_ylabel("Est Rel Bias")
            axes[3].set_xlabel("Truncation Length")
            axes[3].set_yscale('log')


            fig.suptitle("Beta Est Method: {0}".format(beta_estimate_method))
            fig.set_size_inches(8,14)
            fig.savefig(os.path.join(path_to_grad_norm, '{0}.png'.format(
                beta_estimate_method)))

        # Estimate Relative Error + Gradient Norm for all beta methods
        plt.close('all')
        fig, axes = plt.subplots(3,1)

        axes[0].plot(np.arange(1,seq_len+1), grad_norm[:-burnin], color='gray', alpha=0.1)
        for ii, (beta_estimate_method, varphi_hat) in enumerate(beta_log_grad_norm.items()):
            axes[0].plot(np.arange(1, len(varphi_hat)+1),
                    np.exp(varphi_hat),
                    label='{0}'.format(beta_estimate_method), linestyle='--', alpha=0.7)
        axes[0].legend()
        axes[0].set_ylabel("Grad Norm Est")
        axes[0].set_xlabel("Lag")
        axes[0].set_yscale('log')

        axes[1].plot(np.arange(1,seq_len+1), est_abs_bias[:-burnin], color='gray', alpha=0.1)
        for ii, (beta_estimate_method, trun_bias_) in enumerate(beta_trun_abs_bias.items()):
            axes[1].plot(np.arange(1, len(trun_bias_)+1), trun_bias_,
                label='{0}'.format(beta_estimate_method), linestyle='--', alpha=0.7)
        axes[1].legend()
        axes[1].set_ylabel("Est Abs Bias")
        axes[1].set_xlabel("Truncation Length")
        axes[1].set_yscale('log')

        axes[2].plot(np.arange(1,seq_len+1), est_rel_bias[:-burnin], color='gray', alpha=0.1)
        for ii, (beta_estimate_method, trun_bias_) in enumerate(beta_trun_rel_bias.items()):
            axes[2].plot(np.arange(1, len(trun_bias_)+1), trun_bias_,
                label='{0}'.format(beta_estimate_method), linestyle='--', alpha=0.7)
        axes[2].legend()
        axes[2].set_ylabel("Est Rel Bias")
        axes[2].set_xlabel("Truncation Length")
        axes[2].set_yscale('log')

        fig.set_size_inches(8,12)
        fig.savefig(os.path.join(path_to_grad_norm,
            'grad_norm_frac_plot_{0}.png'.format(key)))

        plt.close('all')
    return

def checkpoint_metric_plots(metric_df, path_to_figures, path_to_out):
    metric_df = pd.concat(metric_df, ignore_index=True)
    joblib.dump(metric_df, os.path.join(path_to_out, 'metrics.p.gz'))
    plt.close('all')
    df = metric_df.query('epoch > 1')
    df = df[~df['metric'].str.contains('hidden|cell')] # filter out var specific K
    if df.shape[0] > 0:
        g = sns.FacetGrid(col='metric', col_wrap=4, data=df, sharey=False)
        g.map_dataframe(sns.lineplot, x='epoch', y='value',
                estimator='mean', ci='sd')
        g.fig.savefig(os.path.join(path_to_figures, 'metrics.png'))
        for ax in g.axes.flatten():
            ax.set_xscale('log')
        g.fig.savefig(os.path.join(path_to_figures, 'metrics_logx.png'))
        plt.close('all')
    return

# Loop Setup
pbar = tqdm(range(MAX_TRAIN_STEPS))
exit_flag = False # Whether to exit
cur_train_time = 0.0 # Train Time Elapsed
if len(metric_df) == 0:
    epoch = 0 # Epoch used in training
    adaptive_epoch = 0 # Epoch used in training + adaptive K estimation
    K = args.K # BPTT size
else:
    epoch = metric_df[0].iloc[-1]['epoch']
    adaptive_epoch = metric_df[0].iloc[-1]['adaptive_epoch'] - epoch
    K = int(metric_df[0].query("metric == 'cur_K'").iloc[-1]['value'])

TP_SEQ_LEN = min([TP_SEQ_LEN, train_data.shape[0]-1])
cyclic_init = rnn_module.get_default_init(args.batch_size)
cyclic_index = 0
Khat = K # Mean Estimate for cur_K
deltas = [1.0, 0.9, 0.5, 0.1]
if args.adaptive_K:
    if args.delta not in deltas:
        deltas.append(args.delta)

valid_X, valid_Y = get_batch(valid_data, 0, valid_data.shape[0])
test_X, test_Y = get_batch(test_data, 0, test_data.shape[0])

# Loop
for step in pbar:
    # Scale LR
    lr = args.lr
    if args.scale_lr_K or args.decay_lr:
        if args.decay_lr:
            lr = lr/np.sqrt(step+1)
        if args.scale_lr_K:
            if args.linear_scale:
                lr = lr*K
            else:
                lr = lr*np.sqrt(K)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Training Steps
    start_time = time.time()
    if step > 0:
        step_pbar = range(MAX_STEPS_PER_STEP)
        for step_step in step_pbar:
            # One Step of Optimizer
            if cyclic_index > train_data.shape[0]:
                print("   Epoch {0} Cycle Complete    ".format(int(epoch)))
                cyclic_init = rnn_module.get_default_init(args.batch_size)
                cyclic_index = 0
            tp_seq_len = (TP_SEQ_LEN//K)*K
            partial_X, partial_Y = get_batch(train_data,
                start=cyclic_index, seq_len=tp_seq_len)
            cyclic_index += tp_seq_len
            streaming_train_loss, cyclic_init = runner.train(
                    partial_X, partial_Y, cyclic_init,
                    K=K,
                    hidden_out=True,
                    clip_grad=args.clip_grad,
                    tqdm=tqdm,
                    )
            epoch += partial_X.shape[0]/(train_data.shape[0]-1)
            if time.time() - start_time > MAX_TIME_PER_STEP:
                break
    cur_train_time += time.time() - start_time

    # Compute Metrics
    valid_loss = runner.test(valid_X, valid_Y,
            rnn_module.get_default_init(valid_X.shape[1]),
            K=100, tqdm=tqdm)/valid_Y.size(0)
    test_loss = runner.test(test_X, test_Y,
            rnn_module.get_default_init(test_X.shape[1]),
            K=100, tqdm=tqdm)/test_Y.size(0)

    metric_ests = [None] * NUM_K_EST_REPS
    for est_rep in tqdm(range(NUM_K_EST_REPS), desc="K_est_rep"):
        metric_ests[est_rep] = calc_metric_estimate(train_data, runner,
                cur_K=int(np.round(Khat)),
                seq_len=K_EST_SEQ_LEN, burnin=K_EST_SEQ_LEN//10, tau=K_EST_SEQ_LEN-K_EST_SEQ_LEN//10,
                deltas=deltas,
                )
    relK = int(np.round(np.mean([metric_est['K_0.5'] for metric_est in metric_ests])))
    if args.adaptive_K:
        Khat = np.mean([metric_est['K_{0}'.format(args.delta)] for metric_est in metric_ests])
    pbar.set_description(
            "Epoch: {0:2.1f}, Valid Loss: {1:4.4f}, Test Loss: {2:4.4f}, Test PPL: {3:4.4f}, Cur K: {4:2d}, Khat: {5:3.2f}, K_0.5: {6:2d}, Train Time: {7:4.2f}, LR: {8:4.2f}".format(
            epoch, valid_loss, test_loss, np.exp(test_loss), K, Khat, relK, cur_train_time, lr,
            ))

    metric = [
            dict(metric = 'valid_log_ppl', value = valid_loss),
            dict(metric = 'test_log_loss', value = test_loss),
            dict(metric = 'valid_ppl', value = np.exp(valid_loss)),
            dict(metric = 'test_ppl', value = np.exp(test_loss)),
            dict(metric = 'cur_K', value = K),
            dict(metric = 'Khat', value = Khat),
            ] + [
            dict(metric = key, value=value, rep=rep)
            for rep, metric_est in enumerate(metric_ests)
            for key, value in metric_est.items()
            ]
    metric = pd.DataFrame(metric).fillna(0)
    metric['adaptive_epoch'] = adaptive_epoch + epoch
    metric['epoch'] = epoch
    metric['time'] = cur_train_time
    metric_df.append(metric)

    if (cur_train_time > args.max_train_time) or (epoch > args.epochs):
        exit_flag = True

    # Checkpoints
    if (step % CHECK_FREQ == 0) or exit_flag:
        epoch_str = str(int(epoch*10)/10)
        path_to_figures = os.path.join(args.experiment_folder, 'figures',
                method_name, epoch_str)
        if not os.path.isdir(path_to_figures):
            os.makedirs(path_to_figures)

        print("Quick Checkpoint Epoch:{0}      ".format(epoch_str))
        checkpoint_metric_plots(metric_df, path_to_figures, path_to_out)
        if (step % GRAD_CHECK_FREQ == 0) or exit_flag:
            checkpoint_grad_plots(runner, estK_data, path_to_figures)


    # Exit Early (if max time or epoch exceeded)
    if exit_flag:
        print("Saving Module Parameters to {0}".format(
            path_to_check_param_dict))
        torch.save(rnn_module.state_dict(), path_to_check_param_dict)
        break

    # Update Truncation Size (if adaptive)
    if args.adaptive_K and step > 0:
        start_time = time.time()
        seq_len = K_EST_SEQ_LEN
        tau = seq_len-K_EST_SEQ_LEN//10
        burnin = K_EST_SEQ_LEN//10
        Ks = []
        for _ in range(5):
            K_ests = adaptive_K_est(train_data,
                    runner,
                    seq_len=seq_len,
                    burnin=burnin,
                    tau=tau,
                    deltas=[args.delta],
                    beta_estimate_methods=[args.beta_estimate_method],
                    )
            Ks.append(np.max(list(K_ests.values())))
            adaptive_epoch += (seq_len+burnin)/(train_data.shape[0]-1)
        K = np.max([MIN_K, int(np.mean(Ks))])
        K = np.min([K, MAX_K])
        cur_train_time += time.time() - start_time

print("... Done")
# EOF


