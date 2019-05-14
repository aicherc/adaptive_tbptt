import torch.nn as nn
import numpy as np

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, nout, ninp, nhid, nlayers, dropout=0.2):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Linear(ninp, nhid)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(nhid, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu', 'RNN': 'tanh'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(nhid, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, nout)

        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
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



