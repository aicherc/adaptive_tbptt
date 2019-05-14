""" TBPTT wrapper / source file
"""
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
#import torch.utils.data

import time
from tqdm import tqdm

class TBPTT_minibatch_helper(object):
    """ TBPTT minibatch gradient helper

    Args:
        rnn_module (nn.Module): pytorch module, RNN/SRU/LSTM module
            takes input_seq + state returns output_seq + state
        loss_module (nn.Loss): loss, e.g. nn.CrossEntropyLoss or nn.MSELoss
        optimizer (torch.optim.Optimizer): optimizer, e.g. SGD, Adadelta, etc.
        use_buffer (bool): use right half of sequence as buffer
        original_style (bool): only backprop gradient w.r.t. single loss
        original_buffer (bool): only backprop gradient w.r.t. second half of loss
        double_tf (bool): use 2*K-1 instead of K
        normed (bool): whether to `normalize' TBPTT (default False)
    """
    def __init__(self, rnn_module, loss_module, optimizer,
            use_buffer=False, original_style=False, original_buffer=False,
            double_tf=False, cuda=False, normed=False):
        self.rnn_module = rnn_module
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.use_buffer = use_buffer
        self.original_style = original_style
        self.original_buffer = original_buffer
        self.double_tf = double_tf
        self.cuda = cuda
        self.normed = normed
        return

    def train(self, X, Y, init_state, K,
            tqdm=None, tqdm_message="train pass",
            hidden_out = False, **kwargs):
        """ Split X, Y into subsequences and train over batches

            If use_buffer or original_style, batches will overlap

        Args:
            X (seq_len by batch by input_size torch.Tensor)
            Y (seq_len by batch by output_size torch.Tensor)
            init_state (num_layers by batch by input_size or tuple):
                RNN -> is tensor, for LSTM -> is tuple of tensors
            K (int): subsequence size
            hidden_out (boolean, optional): whether to return final hidden state
            **kwargs (kw args): for optimizer
                clip_grad (None or float): how much to clip gradients

        Returns:
            train_loss (float): loss over the batch

        """
        self.rnn_module.train()
        train_loss = 0.0
        if self.cuda:
            X, Y, init_state = X.cuda(), Y.cuda(), _cuda_state(init_state)
        if K == -1:
            K = X.size(0)
        if self.use_buffer:
            # Split X, Y into overlapping subsequences
            num_subsequences = X.size(0)//K
            pbar = range(num_subsequences)
            if tqdm is not None:
                pbar = tqdm(pbar)
            for i in pbar:
                start = i*K
                end = i*K + K + (K-1)
                buffer_size = min(X.size(0) - end, K-1)
                train_loss_i, init_state = self.train_one_batch(
                        Xbatch = X[start:end],
                        Ybatch = Y[start:end],
                        init_state = _detach_state(init_state),
                        buffer_size = buffer_size,
                        **kwargs,
                        )
                train_loss += train_loss_i
                if tqdm is not None:
                    pbar.set_description("{0} avg loss: {1}".format(
                        tqdm_message, train_loss/(K*(i+1)))
                        )
        elif self.original_buffer:
            # Split X, Y into many overlapping subsequences
            # Could be faster if memory shared among cells
            num_subsequences = X.size(0)//K
            pbar = range(num_subsequences)
            if tqdm is not None:
                pbar = tqdm(pbar)
            for i in pbar:
                start = max(i*K - (K-1), 0)
                end = i*K + K
                burnin_size = end - start - K
                train_loss_i, init_state = self.train_one_batch(
                        Xbatch = X[start:end],
                        Ybatch = Y[start:end],
                        init_state = _detach_state(init_state),
                        burnin_size = burnin_size,
                        **kwargs,
                        )
                train_loss += train_loss_i
                if tqdm is not None:
                    pbar.set_description("{0} avg loss: {1}".format(
                        tqdm_message, train_loss/(K*(i+1)))
                        )
        elif self.original_style:
            # Split X, Y into many overlapping subsequences
            # Could be implemented a lot faster if memory shared among cells
            num_subsequences = X.size(0)-K
            pbar = range(num_subsequences)
            if tqdm is not None:
                pbar = tqdm(pbar)
            for i in pbar:
                start = i
                end = i + K
                burnin_size = K-1
                train_loss_i, init_state = self.train_one_batch(
                        Xbatch = X[start:end],
                        Ybatch = Y[start:end],
                        init_state = _detach_state(init_state),
                        burnin_size = burnin_size,
                        **kwargs,
                        )
                train_loss += train_loss_i
                if tqdm is not None:
                    pbar.set_description("{0} avg loss: {1}".format(
                        tqdm_message, train_loss/(i+1))
                        )
        else:
            if self.double_tf:
                K = 2*K-1
            # Split X, Y into Subsequences
            num_subsequences = X.size(0)//K
            pbar = range(num_subsequences)
            if tqdm is not None:
                pbar = tqdm(pbar)
            for i in pbar:
                start = i*K
                end = (i+1)*K
                train_loss_i, init_state = self.train_one_batch(
                        Xbatch = X[start:end],
                        Ybatch = Y[start:end],
                        init_state = _detach_state(init_state),
                        **kwargs,
                        )
                train_loss += train_loss_i
                if tqdm is not None:
                    pbar.set_description("{0} avg loss: {1}".format(
                        tqdm_message, train_loss/(K*(i+1)))
                        )
        if hidden_out:
            return (train_loss, init_state)
        else:
            return train_loss

    def test(self, X, Y, init_state, K,
            tqdm=None, tqdm_message="test pass",
            **kwargs):
        """ Split X, Y into subsequences and test over batches

        Args:
            X (seq_len by batch by input_size torch.Tensor)
            Y (seq_len by batch by output_size torch.Tensor)
            init_state (num_layers by batch by input_size or tuple):
                RNN -> is tensor, for LSTM -> is tuple of tensors
            K (int): subsequence size
            **kwargs (kw args): for optimizer
                clip_grad (None or float): how much to clip gradients

        Returns:
            test_loss (float): loss over the batch
        """
        self.rnn_module.eval()
        test_loss = 0.0

        if K == -1:
            K = X.size(0)

        if self.cuda:
            X, Y, init_state = X.cuda(), Y.cuda(), _cuda_state(init_state)

        # Split X, Y into Batches
        num_batches = X.size(0)//K
        pbar = range(num_batches)
        if tqdm is not None:
            pbar = tqdm(pbar)
        for i in pbar:
            start = i*K
            end = i*K + K
            test_loss_i, init_state = self.test_one_batch(
                    Xbatch = X[start:end],
                    Ybatch = Y[start:end],
                    init_state = init_state,
                    )
            test_loss += test_loss_i
            if tqdm is not None:
                pbar.set_description("{0} avg loss: {1}".format(
                    tqdm_message, test_loss/(K*(i+1)))
                    )
        return test_loss

    def train_one_batch(self, Xbatch, Ybatch, init_state,
            buffer_size = 0, burnin_size = 0, **kwargs):
        """ Run TBPTT over the subsequences in Xbatch, Ybatch

        Runs one step of the optimizer w.r.t loss over the batch

        Args:
            Xbatch (seq_len by batch by input_size torch.Tensor)
            Ybatch (seq_len by batch by output_size torch.Tensor)
            init_state (num_layers by batch by input_size or tuple):
                RNN -> is tensor, for LSTM -> is tuple of tensors
            **kwargs (kw args): for optimizer
                clip_grad (None or float): how much to clip gradients

        Returns:
            train_loss (float): loss over the batch
            out_state (num_layers by batch by input_size or tuple):
                RNN -> is tensor, for LSTM -> is tuple of tensors
        """
        if buffer_size > 0:
            # Forward Pass
            L = Xbatch.size(0) - buffer_size
            Y_hat, out_state = self.rnn_module(Xbatch[:L], init_state)

            # Buffer
            out_state_ = _detach_state(out_state, requires_grad=True)
            Y_buffer, _ = self.rnn_module(Xbatch[L:], out_state_)
            buffer_loss = torch.zeros(1).cuda() if self.cuda else torch.zeros(1)
            for Y_hat_i, Ybatch_i in zip(Y_buffer, Ybatch[L:]):
                buffer_loss += self.loss_module(Y_hat_i, Ybatch_i)
            buffer_loss.backward()

            # backprop buffer grad on sequence
            if isinstance(out_state_, tuple):
                buffer_grad = tuple([s.grad for s in out_state_])
                self.optimizer.zero_grad()
                for s, grad in zip(out_state, buffer_grad):
                    s.backward(grad, retain_graph=True)
            else:
                buffer_grad = out_state_.grad
                self.optimizer.zero_grad()
                out_state.backward(buffer_grad, retain_graph=True)

            # backprop loss on sequence
            loss = torch.zeros(1).cuda() if self.cuda else torch.zeros(1)
            for Y_hat_i, Ybatch_i in zip(Y_hat, Ybatch[:L]):
                loss += self.loss_module(Y_hat_i, Ybatch_i)
            train_loss = loss.item()
            if self.normed:
                norm = Y_hat.size(0)
                loss = loss/norm
            loss.backward()
            del loss
        elif burnin_size > 0:
            Y_hat_first, out_state = self.rnn_module(
                    Xbatch[:-burnin_size], init_state)
            Y_hat_second, _ = self.rnn_module(Xbatch[-burnin_size:], out_state)
            Y_hat = torch.cat((Y_hat_first, Y_hat_second), dim=0)
            # Use the last K loss functions
            loss = torch.zeros(1).cuda() if self.cuda else torch.zeros(1)
            for Y_hat_i, Ybatch_i in zip(Y_hat[burnin_size:],
                    Ybatch[burnin_size:]):
                loss += self.loss_module(Y_hat_i, Ybatch_i)
            train_loss = loss.item()
            if self.normed:
                norm = Y_hat[burnin_size:].size(0)
                loss = loss/norm
            loss.backward()
            del loss
        else:
            Y_hat, out_state = self.rnn_module(Xbatch, init_state)
            # Use all loss functions
            loss = torch.zeros(1).cuda() if self.cuda else torch.zeros(1)
            for Y_hat_i, Ybatch_i in zip(Y_hat, Ybatch):
                loss += self.loss_module(Y_hat_i, Ybatch_i)
            train_loss = loss.item()
            if self.normed:
                norm = Y_hat.size(0)
                loss = loss/norm
            loss.backward()
            del loss

        # Optimizer Step
        if kwargs.get('clip_grad') is not None:
            clip_grad = kwargs['clip_grad']
            if self.normed:
                clip_grad = clip_grad/norm
            torch.nn.utils.clip_grad_norm_(
                    self.rnn_module.parameters(), clip_grad)
        if kwargs.get('zero_select_gradients') is not None:
            kwargs['zero_select_gradients'](self.rnn_module.named_parameters())

        self.optimizer.step()
        self.optimizer.zero_grad()
        _zero_grad_state(init_state)

        return train_loss, out_state

    def test_one_batch(self, Xbatch, Ybatch, init_state, **kwargs):
        """ Evaluate rnn_module over Xbatch, Ybatch

        Calculate loss over the batch

        Args:
            Xbatch (seq_len by batch by input_size torch.Tensor)
            Ybatch (seq_len by batch by output_size torch.Tensor)
            init_state (num_layers by batch by input_size or tuple):
                RNN -> is tensor, for LSTM -> is tuple of tensors

        Returns:
            test_loss (float): loss over the batch
            out_state (num_layers by batch by input_size or tuple):
                RNN -> is tensor, for LSTM -> is tuple of tensors
        """
        Y_hat, out_state = self.rnn_module(Xbatch, init_state)
        loss = torch.zeros(1)
        if self.cuda:
            loss = loss.cuda()
        for Y_hat_i, Ybatch_i in zip(Y_hat, Ybatch):
            loss += self.loss_module(Y_hat_i, Ybatch_i)
        test_loss = loss.item()
        return test_loss, _detach_state(out_state, requires_grad=True)

    def calculate_all_gradients(self, X, Y, init_state, include_param=False,
            train_mode=True,
            tqdm=None, tqdm_message='calc_all_grads',
            ):
        """ Calculate all gradients

        Args:
            X (seq_len by batch by input_size torch.Tensor): input
            Y (seq_len by batch by output_size torch.Tensor): output
            init_state (torch.Tensor or tuple): init hidden state
                num_layers by batch_size by hidden_size
            include_param (bool): include (all) parameter grads

        Returns:
            grads (dict of ndarray): dict of grads
                key is name of theta
                value is `grad`
                    seq_len by seq_len by batch_size by hidden_dim ndarray
                 or seq_len by seq_len by param_dim ndarray
                `grad[i,j]` is the gradient
                    d state[j] / d theta * d loss[i]/d states[j]
                    averaged over batch + flattened
        """
        if train_mode:
            self.rnn_module.train()
        else:
            self.rnn_module.eval()

        seq_len = X.size(0)
        if seq_len > 1000:
            raise ValueError("You don't want to all this function" +
                    " for large sequences, seq_len= {0} > 1000".format(seq_len))

        grads = {}
        if isinstance(init_state, tuple):
            for layer in range(init_state[0].size(0)):
                grads['hidden{0}'.format(layer)] = \
                        np.zeros((seq_len, seq_len, init_state[0].size(1), init_state[0].size(2)))
        else:
            for layer in range(init_state.size(0)):
                grads['hidden{0}'.format(layer)] = \
                        np.zeros((seq_len, seq_len, init_state.size(1), init_state.size(2)))

        if include_param:
            for (key, value) in self.rnn_module.named_parameters():
                grads[key] = np.zeros(
                        (seq_len, seq_len, value.flatten().size(0)),
                        )

        states = [(None, None)] * (seq_len+1)
        states[0] = (None, init_state)
        pbar = range(seq_len)
        if tqdm is not None:
            pbar = tqdm(pbar, total=seq_len, desc=tqdm_message)
        for j in pbar:
            state = states[j][1]
            state = _detach_state(state, requires_grad=True)

            Y_hat, new_state = self.rnn_module(X[j:j+1], state)
            states[j+1] = (state, new_state)

            # Calculate Gradients for d loss[j]/d h[i]
            self.optimizer.zero_grad()
            _zero_grad_states(states)
            loss = self.loss_module(Y_hat[0], Y[j])
            loss.backward(retain_graph=True)

            for i in range(0, j+1):
                state = states[j-i+1][0]

                # Parameter Gradients
                if include_param:
                    for key, value in self.rnn_module.named_parameters():
                        grads[key][j,j-i] = np.array(value.grad.flatten())
                self.optimizer.zero_grad()

                # Hidden State Gradients
                if isinstance(state, tuple):
                    curr_grad = state[0].grad
                else:
                    curr_grad = state.grad

                for layer, layer_grad in enumerate(curr_grad):
                    grads['hidden{0}'.format(layer)][j,j-i] = \
                            np.array(layer_grad)

                if i == j: break # Exit if end of backprop

                # Backprop Gradient
                prev_state = states[j-i][1]
                if isinstance(state, tuple):
                    for sp, s in zip(prev_state, state):
                        sp.backward(s.grad, retain_graph=True)
                else:
                    prev_state.backward(state.grad, retain_graph=True)

        return grads

    def predict(self, X, init_state):
        """ Return output and states for input X """
        self.rnn_module.eval()
        prev_state = _detach_state(init_state, requires_grad=False)
        Yhat, _ = self.rnn_module(X, init_state)
        Yhat = np.array(Yhat.detach().cpu().numpy())
        return Yhat

def _zero_grad_states(states):
    for (state, new_state) in states:
        if state is not None:
            if isinstance(state, list):
                for state_ii in state:
                    _zero_grad_state(state_ii)
            else:
                _zero_grad_state(state)
        if new_state is not None:
            if isinstance(new_state, list):
                for new_state_ii in new_state:
                    _zero_grad_state(new_state_ii)
            else:
                _zero_grad_state(new_state)
    return

def _zero_grad_state(state):
    if isinstance(state, tuple):
        for state_ii in state:
            if state_ii.grad is not None:
                state_ii.grad.zero_()
    else:
        if state.grad is not None:
            state.grad.zero_()
    return

def _detach_state(state, requires_grad=True):
    if isinstance(state, list):
        return [_detach_state(s, requires_grad=requires_grad)
                for s in state]

    if isinstance(state, tuple):
        state = [s.detach() for s in state]
        for s in state:
            s.requires_grad = requires_grad
        state = tuple(state)
    else:
        state = state.detach()
        state.requires_grad = requires_grad
    return state

def _cuda_state(state):
    if isinstance(state, tuple):
        state = [s.cuda() for s in state]
        state = tuple(state)
    else:
        state = state.cuda()
    return state

