import torch.nn as nn
import numpy as np
import os
from io import open
import torch

# Data Processor
def process_data(path_to_event, path_to_time, time_scale=1.0):
    """ Process event and time data into ndarrays:
        X = [one_hot_Y_j-1 + dt_j-1], Y = [Y_j, dt_j]
    """
    if not os.path.isfile(path_to_event):
        raise ValueError("Path {0} does not exist!".format(path_to_event))
    if not os.path.isfile(path_to_time):
        raise ValueError("Path {0} does not exist!".format(path_to_time))
    events = []
    times = []
    with open(path_to_event, 'r', encoding="utf8") as f:
        for line in f:
            events.append([int(ii) for ii in line.split()])
    with open(path_to_time, 'r', encoding="utf8") as f:
        for line in f:
            times.append([float(ii) for ii in line.split()])

    num_events = np.max([np.max(event) for event in events])
    X = []
    Y = []
    for event, time in zip(events, times):
        e = np.array(event) - 1
        e_one_hot = (np.arange(num_events) == e[...,None]).astype(float)

        t = np.array(time)
        t_diff = t
        t_diff[1:] = (t_diff[1:] - t[:-1]) * time_scale
        t_diff[0] = -1.0

        x = np.zeros((len(event)-1, num_events+1))
        x[:,:num_events] = e_one_hot[:-1]
        x[:,-1] = t_diff[:-1]

        y = np.zeros((len(event)-1, 2))
        y[:,0] = e[1:]
        y[:,1] = t_diff[1:]

        X.append(x)
        Y.append(y)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return X, Y, num_events

def process_data_onehot(path_to_event, path_to_time, time_scale=1.0):
    """ Process event and time data into ndarrays:
        X = [one_hot_Y_j-1, new_t, dt_j-1], Y = [Y_j, dt_j]
    """
    if not os.path.isfile(path_to_event):
        raise ValueError("Path {0} does not exist!".format(path_to_event))
    if not os.path.isfile(path_to_time):
        raise ValueError("Path {0} does not exist!".format(path_to_time))
    events = []
    times = []
    with open(path_to_event, 'r', encoding="utf8") as f:
        for line in f:
            events.append([int(ii) for ii in line.split()])
    with open(path_to_time, 'r', encoding="utf8") as f:
        for line in f:
            times.append([float(ii) for ii in line.split()])

    num_events = np.max([np.max(event) for event in events])
    X = []
    Y = []
    for event, time in zip(events, times):
        e = np.array(event) - 1
        e_one_hot = (np.arange(num_events) == e[...,None]).astype(float)

        t = np.array(time)
        t_diff = np.zeros((t.size,2))
        t_diff[1:,1] = (t[1:] - t[:-1]) * time_scale
        t_diff[0,0] = 1.0

        x = np.zeros((len(event)-1, num_events+2))
        x[:,:num_events] = e_one_hot[:-1]
        x[:,-2:] = t_diff[:-1,:]

        y = np.zeros((len(event)-1, 2))
        y[:,0] = e[1:]
        y[:,1] = t_diff[1:,1]

        X.append(x)
        Y.append(y)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return X, Y, num_events


# RNN Models
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, nout, ninp, emsize, nhid, nlayers=1, dropout=0.0):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.y_encoder = nn.Linear(ninp, emsize)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(emsize+1, nhid+1, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu', 'RNN': 'tanh'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(emsize+1, nhid+1, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.y_decoder = nn.Linear(nhid+1, nout)
        self.log_lambda_decoder = nn.Linear(nhid+1, 1)
        self.t_decoder = nn.Linear(1, 1, bias=False)

        self.init_weights()
        self.rnn_type = rnn_type
        self.emsize = emsize
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, hidden):
        # Input is seq_len, batch, [y_j, d_t_j]
        # Output (decoded) is seq_len, batch, [pred_y_j+1, log_f_t_j+1]
        y_emb = self.drop(self.y_encoder(input[:,:,:-1]))
        emb = torch.cat((y_emb, input[:,:,-1:]), dim=2)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        y_output = output.view(output.size(0)*output.size(1), output.size(2))
        y_decoded = self.y_decoder(y_output)
        log_lambda_decoded = self.log_lambda_decoder(y_output)
        decoded = torch.cat((y_decoded, log_lambda_decoded), dim=1)
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                # Matrices
                val_range = (3.0/np.max(p.shape))**0.5
                p.data.uniform_(-val_range, val_range)
            else:
                # Vectors/Bias
                p.data.zero_()

    def get_default_init(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid+1),
                    weight.new_zeros(self.nlayers, bsz, self.nhid+1))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid+1)

    def get_burnin_init(self, X_burnin):
        batch_size = X_burnin.shape[1]
        default_init = self.get_default_init(batch_size)
        _, burnin_init = self.forward(X_burnin, default_init)
        return burnin_init

class RNNModel_onehot(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, nout, ninp, emsize, nhid, nlayers=1, dropout=0.0):
        super(RNNModel_onehot, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.y_encoder = nn.Linear(ninp-2, emsize)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(emsize+2, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu', 'RNN': 'tanh'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(emsize+2, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.y_decoder = nn.Linear(nhid, nout)
        self.log_lambda_decoder = nn.Linear(nhid, 1)
        self.t_decoder = nn.Linear(1, 1, bias=False)

        self.init_weights()
        self.rnn_type = rnn_type
        self.emsize = emsize
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, hidden):
        # Input is seq_len, batch, [y_j, d_t_j]
        # Output (decoded) is seq_len, batch, [pred_y_j+1, log_f_t_j+1]
        y_emb = self.drop(self.y_encoder(input[:,:,:-2]))
        emb = torch.cat((y_emb, input[:,:,-2:]), dim=2)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        y_output = output.view(output.size(0)*output.size(1), output.size(2))
        y_decoded = self.y_decoder(y_output)
        log_lambda_decoded = self.log_lambda_decoder(y_output)
        decoded = torch.cat((y_decoded, log_lambda_decoded), dim=1)
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                # Matrices
                val_range = (3.0/np.max(p.shape))**0.5
                p.data.uniform_(-val_range, val_range)
            else:
                # Vectors/Bias
                p.data.zero_()

    def get_default_init(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def get_burnin_init(self, X_burnin):
        batch_size = X_burnin.shape[1]
        default_init = self.get_default_init(batch_size)
        _, burnin_init = self.forward(X_burnin, default_init)
        return burnin_init

# Temporal Point Process Loss (see Eq. 14 of Du 2016 KDD)
CEL = nn.CrossEntropyLoss()

def TPP_y_loss(Y_out, Y_target):
    # Y_out is Batch by Events, Y_target is Batch by [y_target, dt_target]
    return CEL(Y_out[:,:-1], Y_target[:,0].long())

def TPP_t_loss(Y_out, Y_target, w):
    # Y_out is Batch by Events, Y_target is Batch by [y_target, dt_target]
    log_lambda_decoded = Y_out[:,-1]
    t_decoded = Y_target[:,1] * w
    log_f_t_decoded = log_lambda_decoded + t_decoded + (
            torch.exp(log_lambda_decoded) - \
            torch.exp(t_decoded+log_lambda_decoded)
            )/w
    return -1.0*torch.mean(log_f_t_decoded)

def TPP_loss(Y_out, Y_target, w):
    # Y_out is Batch by Events, Y_target is Batch by [y_target, dt_target]
    return TPP_y_loss(Y_out, Y_target) + TPP_t_loss(Y_out, Y_target, w)

def TPP_loss_generator(w):
    from functools import partial
    return partial(TPP_loss, w=w)

def zero_one_loss_y_map_predict(Y_out, Y_target):
    # Y_out is Seq_Len by Batch by Events
    # Y_target is Seq_Len by Batch by [y_target, dt_target]
    Y_map_predict = np.argmax(Y_out[:,:,:-1], axis=2)
    return 1.0 - np.mean(Y_map_predict == Y_target[:,:,0].cpu().numpy())

def t_mean_predict(Y_out, Y_target, w, max_dt=None, N=1000):
    # Predict T by Numerical Integration
    # Y_out is Seq_Len by Batch by Events
    # Y_target is Seq_Len by Batch by [y_target, dt_target]
    dt_target = Y_target[:,:,1].cpu().numpy()
    if max_dt is None:
        max_dt = np.max(dt_target)*2

    dt_range = np.linspace(0, np.max(dt_target), N)[:,np.newaxis,np.newaxis] + \
            np.zeros((N, Y_out.shape[0], Y_out.shape[1]))
    log_lambda_decoded = Y_out[:,:,-1][np.newaxis,:,:] + \
            np.zeros((N, Y_out.shape[0], Y_out.shape[1]))
    w = w[0,0].item()

    dt_f_dt = dt_range * np.exp(
        log_lambda_decoded + w*dt_range + (
            np.exp(log_lambda_decoded) - \
            np.exp(log_lambda_decoded + w*dt_range)
            )/w
        )
    mean_dt_predict = np.trapz(dt_f_dt, dt_range, axis=0)
    return mean_dt_predict, dt_target

def rmse_t_mean_predict(Y_out, Y_target, w, **kwargs):
    mean_dt_predict, dt_target = t_mean_predict(Y_out, Y_target, w, **kwargs)
    return np.sqrt(np.mean((mean_dt_predict - dt_target)**2))


