""" Adaptive Truncation Helper """
import numpy as np
import pandas as pd
from scipy.special import logsumexp

import torch
import torch.nn as nn
#import torch.utils.data

import time
from tqdm import tqdm

from .tbptt import _detach_state

EPS = 1e-100

def calculate_gradient_norms(X, Y, init_state, rnn_module, loss_module,
        stride_size=1, cum_grad_norm_approx=False,
        tqdm=None, tqdm_message='calc_grad_norms'):
    """ Return the gradients norms of dL(Yhat[-1], Y[-1])/dH[-1-t]
    where rnn_module takes X[t], H[t-1] -> Yhat[t], H[t]

    Args:
        X (seq_len by batch by input_size torch.Tensor): input
        Y (seq_len by batch by output_size torch.Tensor): output
        init_state (torch.Tensor or tuple): init hidden state
            num_layers by batch by hidden_size
        rnn_module (nn.Module): RNN pytorch module
            takes input_size and hidden_size -> out_size, hidden_size
        loss_module (nn.Loss): loss, e.g. nn.CrossEntropyLoss or nn.MSELoss
        stride_size (int, optional): default 1
            calculation stride (trade memory for speed)

    Returns:
        grad_norm (seq_len by batch ndarray): norms of gradients
            Note: grad_norm[t,s] = || dL_s/dH_[s-t] ||
        cum_grad_norm (seq_len ndarray): cumul gradients norm
            if cum_grad_norm_approx:
                cum_grad_norm[t] = E_s[ sum_{t' <= t} || dL_s/dH_[s-t'] || ]
            else:
                cum_grad_norm[t] = || E_s[ sum_{t' <= t} dL_s/dH_[s-t'] ] ||
    """
    rnn_module.train()
    seq_len = (X.size(0) // stride_size)*stride_size
    batch = X.size(1)
    if seq_len > 1000:
        raise ValueError("You don't want to all this function" +
                " for large sequences, seq_len= {0} > 1000".format(seq_len))

    if isinstance(init_state, tuple):
        # For LSTM (hidden_state, cell_state)
        num_layers = init_state[0].size(0)
        hidden_size = init_state[0].size(2)
        flat_grad = np.zeros((batch, 2*num_layers*hidden_size))
        cum_grad = np.zeros((2*num_layers*hidden_size))
    else:
        num_layers = init_state.size(0)
        hidden_size = init_state.size(2)
        flat_grad = np.zeros((batch, num_layers*hidden_size))
        cum_grad = np.zeros((num_layers*hidden_size))

    grad_norm = np.zeros((seq_len, batch))
    cum_grad_norm = np.zeros((seq_len))


    states = [[None]*(stride_size+1) for _ in range(seq_len//stride_size+1)]
    states[0][-1] = init_state

    # Forward Pass
    pbar = range(seq_len//stride_size)
    if tqdm is not None:
        pbar = tqdm(pbar, total=seq_len, desc=tqdm_message)
    for j in pbar:
        state = states[j][-1]
        state = _detach_state(state, requires_grad=True)
        states[j+1][0] = state
        for l in range(stride_size):
            t = j * stride_size + l
            Y_hat, state = rnn_module(X[t:t+1], state)
            states[j+1][l+1] = state

    # Calculate Gradients for d loss[-1]/d h[i]
    loss = loss_module(Y_hat[-1], Y[seq_len-1])
    loss.backward(retain_graph=True)
    # Backprop
    for i in range(seq_len//stride_size):
        for l in range(stride_size):
            t = seq_len-i*stride_size-l-1
            state = states[-i-1][-l-2]
            # Hidden State Gradients
            if isinstance(state, tuple):
                # LSTM
                curr_grad = state[0].grad
                curr_cell_grad = state[1].grad
                for layer, layer_grad in enumerate(curr_grad):
                    flat_grad[:,layer*hidden_size:(layer+1)*hidden_size] = \
                            np.array(layer_grad.cpu())
                for layer, layer_grad in enumerate(curr_cell_grad):
                    flat_grad[:,(num_layers+layer)*hidden_size:(num_layers+layer+1)*hidden_size] = \
                            np.array(layer_grad.cpu())
            else:
                curr_grad = state.grad
                for layer, layer_grad in enumerate(curr_grad):
                    flat_grad[:,layer*hidden_size:(layer+1)*hidden_size] = \
                            np.array(layer_grad.cpu())

            grad_norm[-1-t] = np.sqrt(np.sum(flat_grad**2, axis=1))

            cum_grad += np.mean(flat_grad, axis=0)
            cum_grad_norm[-1-t] = np.sqrt(np.sum(cum_grad**2))

        if i == (seq_len//stride_size-1): break # Exit if end of backprop

        # Backprop Gradient
        prev_state = states[-i-2][-1]
        if isinstance(state, tuple):
            for sp, s in zip(prev_state, state):
                sp.backward(s.grad, retain_graph=True)
        else:
            prev_state.backward(state.grad, retain_graph=True)

    if cum_grad_norm_approx:
        cum_grad_norm = np.cumsum(np.mean(grad_norm, axis=1))
    return grad_norm, cum_grad_norm

def calculate_component_gradient_norms(X, Y, init_state, rnn_module, loss_module,
        stride_size=1, cum_grad_norm_approx=False,
        tqdm=None, tqdm_message='calc_grad_norms'):
    """ Return the gradients norms of dL(Yhat[-1], Y[-1])/dH[-1-t]
    where rnn_module takes X[t], H[t-1] -> Yhat[t], H[t]

    Returns the gradients split by layer/cell

    Args:
        X (seq_len by batch by input_size torch.Tensor): input
        Y (seq_len by batch by output_size torch.Tensor): output
        init_state (torch.Tensor or tuple): init hidden state
            num_layers by batch by hidden_size
        rnn_module (nn.Module): RNN pytorch module
            takes input_size and hidden_size -> out_size, hidden_size
        loss_module (nn.Loss): loss, e.g. nn.CrossEntropyLoss or nn.MSELoss
        stride_size (int, optional): default 1
            calculation stride (trade memory for speed)

    Returns:
        grad_norms (num_component by seq_len by batch ndarray): norms of gradients
            Note: grad_norm[t,s] = || dL_s/dH_[s-t] ||
        cum_grad_norms (num_component by seq_len ndarray): cumul gradients norms
            cum_grad_norm = cum_grad_norms[component]
            if cum_grad_norm_approx:
                cum_grad_norm[t] = E_s[ sum_{t' <= t} || dL_s/dH_[s-t'] || ]
            else:
                cum_grad_norm[t] = || E_s[ sum_{t' <= t} dL_s/dH_[s-t'] ] ||
        component_names (num_component list of strings)
    """
    rnn_module.train()
    seq_len = (X.size(0) // stride_size)*stride_size
    batch = X.size(1)
    if seq_len > 1000:
        raise ValueError("You don't want to all this function" +
                " for large sequences, seq_len= {0} > 1000".format(seq_len))

    if isinstance(init_state, tuple):
        # For LSTM (hidden_state, cell_state)
        num_layers = init_state[0].size(0)
        hidden_size = init_state[0].size(2)
        flat_grad = np.zeros((2*num_layers, batch, hidden_size))
        cum_grad = np.zeros((2*num_layers, hidden_size))
    else:
        num_layers = init_state.size(0)
        hidden_size = init_state.size(2)
        flat_grad = np.zeros((num_layers, batch, hidden_size))
        cum_grad = np.zeros((num_layers, hidden_size))

    # Will swap 0 + 1 axes before returning
    grad_norms = np.zeros((seq_len, flat_grad.shape[0], batch))
    cum_grad_norms = np.zeros((seq_len, flat_grad.shape[0]))


    states = [[None]*(stride_size+1) for _ in range(seq_len//stride_size+1)]
    states[0][-1] = init_state

    # Forward Pass
    pbar = range(seq_len//stride_size)
    if tqdm is not None:
        pbar = tqdm(pbar, total=seq_len, desc=tqdm_message)
    for j in pbar:
        state = states[j][-1]
        state = _detach_state(state, requires_grad=True)
        states[j+1][0] = state
        for l in range(stride_size):
            t = j * stride_size + l
            Y_hat, state = rnn_module(X[t:t+1], state)
            states[j+1][l+1] = state

    # Calculate Gradients for d loss[-1]/d h[i]
    loss = loss_module(Y_hat[-1], Y[seq_len-1])
    loss.backward(retain_graph=True)
    # Backprop
    for i in range(seq_len//stride_size):
        for l in range(stride_size):
            t = seq_len-i*stride_size-l-1
            state = states[-i-1][-l-2]
            # Hidden State Gradients
            if isinstance(state, tuple):
                # LSTM
                curr_grad = state[0].grad
                curr_cell_grad = state[1].grad
                for layer, layer_grad in enumerate(curr_grad):
                    flat_grad[layer] = np.array(layer_grad.cpu())
                for layer, layer_grad in enumerate(curr_cell_grad):
                    flat_grad[num_layers+layer,:,:] = np.array(layer_grad.cpu())
            else:
                curr_grad = state.grad
                for layer, layer_grad in enumerate(curr_grad):
                    flat_grad[layer] = np.array(layer_grad.cpu())

            grad_norms[-1-t] = np.sqrt(np.sum(flat_grad**2, axis=2))

            cum_grad += np.mean(flat_grad, axis=1)
            cum_grad_norms[-1-t] = np.sqrt(np.sum(cum_grad**2, axis=1))

        if i == (seq_len//stride_size-1): break # Exit if end of backprop

        # Backprop Gradient
        prev_state = states[-i-2][-1]
        if isinstance(state, tuple):
            for sp, s in zip(prev_state, state):
                sp.backward(s.grad, retain_graph=True)
        else:
            prev_state.backward(state.grad, retain_graph=True)

    # Reorder
    grad_norms = np.swapaxes(grad_norms, 0, 1)
    cum_grad_norms = np.swapaxes(cum_grad_norms, 0, 1)

    # Labels
    component_names = ['hidden{0}'.format(ii) for ii in range(num_layers)]
    if grad_norms.shape[0] > num_layers:
        component_names += ['cell{0}'.format(ii) for ii in range(num_layers)]

    if cum_grad_norm_approx:
        cum_grad_norms = np.array([np.cumsum(np.mean(grad_norm, axis=1))
            for grad_norm in grad_norms])

    return grad_norms, cum_grad_norms, component_names

def calculate_elementwise_gradients(X, Y, init_state, rnn_module, loss_module,
        stride_size=1, tqdm=None, tqdm_message='calc_grad_norms',
        include_param=False):
    """ Return the gradients norms of dL(Yhat[-1], Y[-1])/dH[-1-t]
    where rnn_module takes X[t], H[t-1] -> Yhat[t], H[t]

    Args:
        X (seq_len by batch by input_size torch.Tensor): input
        Y (seq_len by batch by output_size torch.Tensor): output
        init_state (torch.Tensor or tuple): init hidden state
            num_layers by batch by hidden_size
        rnn_module (nn.Module): RNN pytorch module
            takes input_size and hidden_size -> out_size, hidden_size
        loss_module (nn.Loss): loss, e.g. nn.CrossEntropyLoss or nn.MSELoss
        stride_size (int, optional): default 1
            calculation stride (trade memory for speed)
        include_param (bool, optional): whether to include parameter grads
            (note this is slow)

    Returns:
        grads (dict of seq_len by batch by hidden_size ndarray): norms of gradients
    """
    rnn_module.train()
    seq_len = (X.size(0) // stride_size)*stride_size
    batch = X.size(1)
    if seq_len > 1000:
        raise ValueError("You don't want to all this function" +
                " for large sequences, seq_len= {0} > 1000".format(seq_len))
    grads = {}
    if isinstance(init_state, tuple):
        # For LSTM (init_state is (hidden_state, cell_state)
        for layer in range(init_state[0].size(0)):
            grads['hidden{0}'.format(layer)] = np.zeros((seq_len, batch, init_state[0].size(2)))
            grads['cell{0}'.format(layer)] = np.zeros((seq_len, batch, init_state[0].size(2)))
    else:
        for layer in range(init_state.size(0)):
            grads['hidden{0}'.format(layer)] = np.zeros((seq_len, batch, init_state.size(2)))

    if include_param:
        for (key, value) in rnn_module.named_parameters_of_interest():
            grads[key] = np.zeros((seq_len, batch, value.numel()))

    states = [[None]*(stride_size+1) for _ in range(seq_len//stride_size+1)]
    states[0][-1] = init_state

    pbar = range(seq_len//stride_size)
    if tqdm is not None:
        pbar = tqdm(pbar, total=seq_len, desc=tqdm_message)
    for j in pbar:
        state = states[j][-1]
        state = _detach_state(state, requires_grad=True)
        states[j+1][0] = state
        for l in range(stride_size):
            t = j * stride_size + l
            Y_hat, state = rnn_module(X[t:t+1], state)
            states[j+1][l+1] = state


    # Calculate Gradients for d loss[-1]/d h[i]
    if not include_param: # Do Everying Batchwise if only w.r.t hidden state
        loss = loss_module(Y_hat[-1], Y[seq_len-1])
        loss.backward(retain_graph=True)
        # Backprop
        for i in range(seq_len//stride_size):
            for l in range(stride_size):
                t = seq_len-i*stride_size-l-1
                state = states[-i-1][-l-2]
                # Hidden State Gradients
                if isinstance(state, tuple):
                    curr_grad = state[0].grad
                    curr_cell_grad = state[1].grad
                    for layer, layer_grad in enumerate(curr_grad):
                        grads['hidden{0}'.format(layer)][-1-t] = layer_grad
                    for layer, layer_grad in enumerate(curr_cell_grad):
                        grads['cell{0}'.format(layer)][-1-t] = layer_grad
                else:
                    curr_grad = state.grad
                    for layer, layer_grad in enumerate(curr_grad):
                        grads['hidden{0}'.format(layer)][-1-t] = layer_grad

            if i == (seq_len//stride_size-1): break # Exit if end of backprop

            # Backprop Gradient
            prev_state = states[-i-2][-1]
            if isinstance(state, tuple):
                for sp, s in zip(prev_state, state):
                    sp.backward(s.grad, retain_graph=True)
            else:
                prev_state.backward(state.grad, retain_graph=True)
    else:
        pbar = range(batch)
        if tqdm is not None:
            pbar = tqdm(pbar, total=batch, desc=tqdm_message+"_backprop")
        for b in pbar:
            loss = loss_module(Y_hat[-1,b:b+1], Y[seq_len-1,b:b+1])
            loss.backward(retain_graph=True)
            # Backprop
            for i in range(seq_len//stride_size):
                for l in range(stride_size):
                    t = seq_len-i*stride_size-l-1
                    state = states[-i-1][-l-2]
                    # Hidden State Gradients
                    if isinstance(state, tuple):
                        curr_grad = state[0].grad
                        curr_cell_grad = state[1].grad
                        for layer, layer_grad in enumerate(curr_grad):
                            grads['hidden{0}'.format(layer)][-1-t,b] = layer_grad[b]
                        for layer, layer_grad in enumerate(curr_cell_grad):
                            grads['cell{0}'.format(layer)][-1-t,b] = layer_grad[b]
                    else:
                        curr_grad = state.grad
                        for layer, layer_grad in enumerate(curr_grad):
                            grads['hidden{0}'.format(layer)][-1-t,b] = layer_grad[b]

                    # Parameters Gradients
                    if include_param:
                        for key, value in rnn_module.named_parameters_of_interest():
                            if value.grad is not None:
                                grads[key][-1-t,b] = value.grad.flatten()
                                value.grad.data.zero_() # zero gradient

                if i == (seq_len//stride_size-1): break # Exit if end of backprop
                # Backprop Gradient
                prev_state = states[-i-2][-1]
                if isinstance(state, tuple):
                    for sp, s in zip(prev_state, state):
                        sp.backward(s.grad, retain_graph=True)
                        s.grad.data.zero_() # zero gradient
                else:
                    prev_state.backward(state.grad, retain_graph=True)
                    state.grad.data.zero_() # zero gradient

    return grads

def estimate_logbeta_max(log_grad_norm, tau, W=None):
    """ Estimate logbeta using varphi[tau_hat:W, :]

    Args:
        log_grad_norm (seq_len by batch): gradient norms
        tau (int): estimate for when decay starts
        W (int): end of window used to estimate logbeta
    """
    seq_len = log_grad_norm.size
    if W is None:
        W = seq_len-1
    # Check Size
    if (W >= seq_len) or (tau > W):
        raise ValueError("tau = {0} < W = {1} < grad_norm.shape[0]={2}".format(
            tau, W, seq_len))

    logbeta = np.max(
            (log_grad_norm[tau+1:W+1] - log_grad_norm[tau]) / np.arange(1, W-tau+1)
        )
    return logbeta

def estimate_logbeta_quantile(log_grad_norm, tau, W=None, quantile=0.95):
    """ Estimate logbeta using varphi[tau_hat:W, :]

    Args:
        log_grad_norm (seq_len by batch): gradient norms
        tau (int): estimate for when decay starts
        W (int): end of window used to estimate logbeta
        quantile (float): quantile in (0,1) (default is 0.95)
    """
    seq_len = log_grad_norm.size
    if W is None:
        W = seq_len-1
    # Check Size
    if (W >= seq_len) or (tau > W):
        raise ValueError("tau = {0} < W = {1} < grad_norm.shape[0]={2}".format(
            tau, W, seq_len))

    diff = np.subtract.outer(log_grad_norm[tau+1:W+1], log_grad_norm[tau:W])
    denom = np.subtract.outer(np.arange(1,W-tau+1), np.arange(0,W-tau))
    diff = diff[np.tril_indices_from(diff)]
    denom = denom[np.tril_indices_from(denom)]

    logbeta = np.quantile(diff/denom, quantile)
    return logbeta

def estimate_logbeta_ols(log_grad_norm, tau, W=None):
    """ Estimate logbeta using varphi[tau_hat:W, :]

    Args:
        grad_norm (seq_len by batch): gradient norms
        tau (int): estimate for when decay starts
        W (int): end of window used to estimate logbeta
    """
    seq_len = log_grad_norm.size
    if W is None:
        W = seq_len-1
    # Check Size
    if (W >= seq_len) or (tau > W):
        raise ValueError("tau = {0} < W = {1} < grad_norm.shape[0]={2}".format(
            tau, W, seq_len))

    Yraw = log_grad_norm + 0.0
    Xraw = np.arange(0, Yraw.shape[0], dtype=float)

    Xraw[Yraw < -50] = np.NaN # Values less than -50 are essentially zero
    Yraw[Yraw < -50] = np.NaN # Values less than -50 are essentially zero

    if np.sum(~np.isnan(Yraw)) <= 1:
        # Case when most values are less than -50
        logbeta = 0
        return logbeta

    # Center
    Y = Yraw - np.nanmean(Yraw)
    X = Xraw - np.nanmean(Xraw)

    # Estimate (X^T X)^{-1} (X^T Y)
    logbeta = np.nansum(X*Y)/np.nansum(X**2)
    return logbeta

def log_abs_error_estimate(log_grad_norm, logbeta, tau, trun_range=None):
    """ Estimate the Expected Absolute Bias in truncations at trun_range """
    seq_len = log_grad_norm.size
    if tau >= seq_len:
        raise ValueError("tau = {0} must be < grad_norm.shape[0]={1}".format(
            tau, seq_len))
    if trun_range is None:
        trun_range = np.arange(seq_len)+1


    logsumexp_log_grad_norm = np.zeros(tau+1, dtype=float)
    logsumexp_log_grad_norm[tau] = log_grad_norm[tau]
    for t in reversed(range(0,tau)):
        logsumexp_log_grad_norm[t] = np.logaddexp(
                logsumexp_log_grad_norm[t+1],
                log_grad_norm[t]
                )

    log_abs_error = np.zeros_like(trun_range, dtype=float)
    if logbeta < 0:
        log_abs_error[trun_range < tau] = np.logaddexp(
                logsumexp_log_grad_norm[trun_range[trun_range < tau]],
                log_grad_norm[tau] - np.log(1-np.exp(logbeta))
                )
        log_abs_error[trun_range >= tau] = (
                log_grad_norm[tau] +
                (trun_range[trun_range>=tau]-tau)*logbeta -
                np.log(1-np.exp(logbeta))
                )
    elif logbeta == 0 and log_grad_norm[tau] < -50:
        log_abs_error[trun_range < tau] = \
                logsumexp_log_grad_norm[trun_range[trun_range < tau]]
        log_abs_error[trun_range >= tau] = -50
    else:
        log_abs_error[trun_range >= tau] = np.inf


    return log_abs_error

def log_lower_total_grad_norm_estimate(
        log_cum_grad_norm, log_grad_norm, logbeta, tau):
    """ Estimate a lower bound for || g(theta) || """
    trun_range = np.arange(0, log_grad_norm.shape[0])
    log_abs_error = log_abs_error_estimate(
            log_grad_norm, logbeta, tau, trun_range)

    log_lower_total_grad_norm_bounds = np.zeros_like(trun_range, dtype=float)

    bound = log_cum_grad_norm > log_abs_error
    # Bound with \| g_K \| - abs_error(K)
    log_lower_total_grad_norm_bounds[bound] = log_cum_grad_norm[bound] + np.log(
            1.0 - np.exp(log_abs_error[bound] - log_cum_grad_norm[bound]))
    log_lower_total_grad_norm_bounds[~bound] = -np.inf

    log_lower_total_grad_norm_bound = np.max(log_lower_total_grad_norm_bounds)
    return log_lower_total_grad_norm_bound

def log_rel_error_estimate(log_cum_grad_norm, log_grad_norm, logbeta, tau,
        trun_range=None,
        log_abs_error=None, log_lower_total_grad_norm_bound=None):
    """ Estimate the Expected Relative Bias in truncations at trun_range """
    seq_len = log_grad_norm.size
    if tau >= seq_len:
        raise ValueError("tau = {0} must be < grad_norm.shape[0]={1}".format(
            tau, seq_len))

    if trun_range is None:
        trun_range = np.arange(seq_len)+1

    # Calculate Numerator: Upper Bound on Expected Abs Error
    if log_abs_error is None:
        log_abs_error = log_abs_error_estimate(
                log_grad_norm, logbeta, tau, trun_range=None,
                )
    elif len(log_abs_error) != len(trun_range):
        raise ValueError("log_abs_error should be same size as trun_range")

    # Calculate Lower Bound on ||g(\theta)||
    if log_lower_total_grad_norm_bound is None:
        log_lower_total_grad_norm_bound = log_lower_total_grad_norm_estimate(
                log_cum_grad_norm, log_grad_norm, logbeta, tau,
                )

    # Relative Error Bound
    log_rel_error = log_abs_error - log_lower_total_grad_norm_bound
    return log_rel_error

def adaptive_K_estimate(grad_norm, cum_grad_norm, tau, deltas,
        trun_range=None, estimator=None, est_after_log=False,
        beta_estimate_method='ols', W=None,
        extra_out=False, **kwargs):
    """ Adaptive Estimation of K

    Select Smallest K such that the estimated expected relative error < delta

    Args:
        grad_norm (seq_len by batch): grad norms (see calculate_gradient_norms)
        cum_grad_norm (seq_len): cumulative grad norms
        tau (int): guess for when exponential decay kicks-in
        deltas (ndarray of float): target relative error bounds
        trun_range (ndarray of ints): range of potential truncations
            (default is [1,seq_len])
        estimator (func): how to average over the minibatch (default np.mean)
        est_after_log (bool): whether to average before or after applying log
        beta_estimate_method (str): one of 'max', 'quantile', 'ols'
        W (int): end of window used to estimate logbeta
        extra_out (bool): whether to return extra output
    Returns:
        K_est (ndarray of int) best K
    Or if extra_out is True:
        dict of
            K_est (ndarray) best K for each delta in deltas
            log_rel_error (ndarray) log relative error for trun_range
            log_abs_error (ndarray) log absolute error for trun_range
            logbeta (double) estimate given grad_norm, cum_grad_norm, and tau
    """
    seq_len, batch = grad_norm.shape
    if tau >= seq_len:
        raise ValueError("tau = {0} must be < grad_norm.shape[0]={1}".format(
            tau, seq_len))

    if trun_range is None:
        trun_range = np.arange(seq_len)+1

    # Take Expectation of grad_norm
    if estimator is None:
        estimator = np.mean
    if est_after_log:
        log_grad_norm = estimator(np.log(grad_norm+EPS), axis=1)
    else:
        log_grad_norm = np.log(estimator(grad_norm, axis=1)+EPS)
    log_cum_grad_norm = np.log(cum_grad_norm+EPS)

    # Estimate Beta
    if beta_estimate_method == 'ols':
        logbeta = estimate_logbeta_ols(log_grad_norm, tau, W)
    elif beta_estimate_method == 'max':
        logbeta = estimate_logbeta_max(log_grad_norm, tau, W)
    elif beta_estimate_method == 'quantile':
        logbeta = estimate_logbeta_quantile(log_grad_norm, tau, W,
                **kwargs)
    else:
        raise ValueError("Unrecognized beta_estimate_method {0}".format(
            beta_estimate_method))

    # Estimate Abs Error
    log_abs_error = log_abs_error_estimate(
        log_grad_norm=log_grad_norm,
        logbeta=logbeta,
        tau=tau,
        trun_range=trun_range)

    # Estimate Rel Error
    log_rel_error = log_rel_error_estimate(
        log_cum_grad_norm=log_cum_grad_norm,
        log_grad_norm=log_grad_norm,
        logbeta=logbeta,
        tau=tau,
        trun_range=trun_range,
        log_abs_error=log_abs_error)

    # Select K
    K_est = np.zeros_like(deltas, dtype=int)
    for ii, delta in enumerate(deltas):
        if np.all(log_rel_error > np.log(delta+EPS)):
            # If Relative Error is not bounded return the largest K
            K_est[ii] = trun_range[-1]
        else:
            # Otherwise Return the Smallest K Relative Error is bounded
            K_est[ii] = trun_range[
                np.argmax(log_rel_error < np.log(delta+EPS))
                ]

    # Output
    if extra_out:
        return dict(
            K_est=K_est,
            logbeta=logbeta,
            log_abs_error=log_abs_error,
            log_rel_error=log_rel_error,
            )
    else:
        return K_est

def log_estimate_grad_norm(grad_norm, logbeta, tau, trun_range=None,
        estimator=None, est_after_log=False, **kwargs):
    """ Helper function for estimate the grad norm using beta+tau"""
    seq_len, batch = grad_norm.shape
    if tau >= seq_len:
        raise ValueError("tau = {0} must be < grad_norm.shape[0]={1}".format(
            tau, seq_len))

    if trun_range is None:
        trun_range = np.arange(seq_len)+1

    # Take Expectation of grad_norm
    if estimator is None:
        estimator = np.mean
    if est_after_log:
        log_grad_norm = estimator(np.log(grad_norm+EPS), axis=1)
    else:
        log_grad_norm = np.log(estimator(grad_norm, axis=1)+EPS)

    logsumexp_log_grad_norm = np.zeros(seq_len, dtype=float)
    logsumexp_log_grad_norm[0] = log_grad_norm[0]
    for t in range(1,seq_len):
        logsumexp_log_grad_norm[t] = np.logaddexp(
                logsumexp_log_grad_norm[t-1],
                log_grad_norm[t],
                )

    log_est_norm = np.zeros_like(trun_range, dtype=float)
    log_est_norm[trun_range <= tau] = \
            log_grad_norm[trun_range[trun_range <= tau]]
    log_est_norm[trun_range > tau] = log_grad_norm[tau] + \
            logbeta*(trun_range[trun_range>tau] - tau)
    return log_est_norm






