""" Sequence Generator
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
#import torch.utils.data

import time
from tqdm import tqdm

def generate_linear_seq(seq_len, batch_size, input_size, output_size,
        A=None, B=None, C=None, D=None, latent_size=None,
        noise_sd=0.01, lag=1, **kwargs):
    """ Generate LDS Control Sequence """
    if latent_size is None:
        latent_size = output_size
    if np.shape(A) != (latent_size, latent_size):
        raise ValueError("A shape is wrong")
    if np.shape(B) != (latent_size, input_size):
        raise ValueError("B shape is wrong")
    if np.shape(C) != (output_size, input_size):
        raise ValueError("C shape is wrong")
    if np.shape(D) != (output_size, latent_size):
        raise ValueError("D shape is wrong")

    input_seq = np.zeros((seq_len, batch_size, input_size))
    latent_seq = np.zeros((seq_len, batch_size, latent_size))
    output_seq = np.zeros((seq_len, batch_size, output_size))

    latent_prev = np.zeros((batch_size, latent_size))
    for t in range(seq_len):
        input_seq[t] = np.random.normal(size=(batch_size, input_size))

    for t in range(seq_len):
        latent_seq[t] = latent_prev.dot(A.T) + input_seq[t-lag].dot(B.T)
        output_seq[t] = latent_seq[t].dot(C.T) + input_seq[t-lag].dot(D.T) + \
                np.random.normal(scale=noise_sd, size=(batch_size, output_size))
        latent_prev = latent_seq[t]

    return dict(
            input_seq=input_seq,
            latent_seq=latent_seq,
            output_seq=output_seq,
            )

def generate_repeating_sequence(seq_len, batch_size, input_size, output_size,
        grow=True, lag=1, base_seq_length=10, **kwargs):
    """ Generate Repeat Input + Output Sequence

    Args:
        seq_len (int): length of sequences
        batch_size (int): number of sequences
        input_size (int): number of states / input one-hot encoding dim
        output_size (int): not used, same as input_size
        lag (int): lag before repeating input sequence
        base_seq_length (int, optional) how long sequence is before repeating
        grow (bool, optional): add 1 modulo num_states to repeating sequence

    Returns: (as dict)
        input_seq (seq_len by batch_size by num_states ndarray) -> one-hot
        output_seq (seq_len by batch_size ndarray) -> indices

    Example: (with lag 2)
        input_seq  = [2, 4, 1, 2, 3, 4, 5] (one hot encoding)
        output_seq = [?, ?, 2, 4, 1, 2, 3]
    """
    num_states = input_size
    seq_copies = int(seq_len//base_seq_length)+1

    base_sequence = np.random.multinomial(n=1,
            pvals=np.ones(num_states)/num_states,
            size=(base_seq_length, batch_size))
    if grow:
        sequence = np.array([
            np.concatenate(
                (base_sequence[:,:,-(ii%num_states):],
                 base_sequence[:,:,:-(ii%num_states)]),
                axis=2)
            for ii in range(seq_copies)])
    else:
        sequence = np.array([base_sequence for _ in range(seq_copies)])
    sequence = np.reshape(sequence, (-1, batch_size, num_states))
    input_sequence = sequence[lag:seq_len+lag]
    output_sequence = sequence[:seq_len].dot(np.arange(num_states))
    return dict(
            input_seq=input_sequence,
            output_seq=output_sequence,
            )

def generate_copy_sequence(seq_len, batch_size, num_states, lag, min_lag=None,
        **kwargs):
    """ Generate Copy Input + Output Sequence

    Args:
        seq_len (int): length of sequences
        batch_size (int): number of sequences
        num_states (int): number of states (with states = 0 + 1 reserved)
            (num_states should be greater than 2)
        lag (int): max_length of copy input sequence
        min_lag (int): min_length of copy input sequence, default lag/2

    Returns: (as dict)
        input_seq (seq_len by batch_size by num_states ndarray) -> one-hot
        output_seq (seq_len by batch_size ndarray) -> indices

    Example:
        input_seq  = [ # 2 3 4 - - - - # 4 3 5 2 4 - - - - - -]
        output_seq = [ - - - - # 2 3 4 - - - - - - # 4 3 5 2 4]
    """
    if min_lag is None:
        min_lag = np.max([lag//2, 1])
    if num_states < 2:
        raise ValueError("num_states should be greater than 2")
    input_seq = np.zeros((seq_len+4*lag, batch_size), dtype=int)
    output_seq = np.zeros((seq_len+4*lag, batch_size), dtype=int)
    for batch in range(batch_size):
        index = 0
        while index < seq_len+2*lag:
            copy_length = np.random.randint(min_lag, lag+1)
            copy_seq = np.random.randint(2, num_states, size=copy_length)
            copy_seq[0] = 1
            input_seq[index:index+copy_length, batch] = copy_seq
            output_seq[index+copy_length:index+2*copy_length, batch] = copy_seq
            index += 2*copy_length
        # shift input_seq + output_seq by a random amount
        random_shift = np.random.randint(0, 2*lag)
        input_seq[:seq_len,:] = input_seq[random_shift:seq_len+random_shift,:]
        output_seq[:seq_len,:] = output_seq[random_shift:seq_len+random_shift,:]

    # Convert
    a = input_seq[:seq_len,:]
    input_seq_one_hot = (np.arange(a.max()+1) == a[...,None]).astype(int)
    output_seq = output_seq[:seq_len]
    return dict(
            input_seq=input_seq_one_hot,
            output_seq=output_seq,
            )

def generate_input_sequence(parameters, generate_seq,
        seq_len, batch_size, layer_size, hidden_size, **kwargs):
    """ generate_seq must return a dict with input_seq and output_seq """
    data = generate_seq(seq_len=seq_len, batch_size=batch_size,
            input_size=layer_size, output_size=layer_size,
            **parameters)
    X = torch.tensor(data['input_seq'], dtype=torch.float32)
    if np.issubdtype(data['output_seq'].dtype, np.integer):
        Y = torch.tensor(data['output_seq'], dtype=torch.long)
    else:
        Y = torch.tensor(data['output_seq'], dtype=torch.float32)
    init_latent = torch.zeros(batch_size, hidden_size)
    return data, X, Y, init_latent


def plot_sequence(Yhat, Ytrue, color=None):
    """ Yhat is (seq_len, num_states), Ytrue = (seq_len) """
    import matplotlib.pyplot as plt
    seq_len, num_states = Yhat.shape
    if color is None:
        if num_states < 8:
            color = ['C{0}'.format(ii) for ii in range(num_states)]
        else:
            import seaborn as sns
            color = sns.color_palette('husl', num_states)

    fig, ax = plt.subplots(1,1)
    Yprob = np.exp(Yhat) / np.outer(np.sum(np.exp(Yhat), axis=1),
            np.ones(num_states))

    for ii in range(num_states):
        ax.plot(Yprob[:,ii], color=color[ii])
    for ii in range(num_states):
        ax.plot(np.arange(seq_len)[Ytrue == ii], np.ones(np.sum(Ytrue==ii)),
            'o', color=color[ii])
    return fig, ax


def plot_repeating_sequence(Yhat, Ytrue):
    """ Yhat is (seq_len, num_states), Ytrue = (seq_len) """
    # Depreciated Function in favor of plot sequence
    return plot_sequence(Yhat, Ytrue)

def plot_copy_sequence(Yhat, Ytrue):
    """ Yhat is (seq_len, num_states), Ytrue = (seq_len) """
    seq_len, num_states = Yhat.shape
    color = ['C{0}'.format(ii-2) for ii in range(num_states)]
    color[0] = 'w'
    color[1] = 'k'
    return plot_sequence(Yhat, Ytrue, color=color)




