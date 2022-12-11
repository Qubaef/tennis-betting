import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        target_size,
        num_layers,
        batch_size,
        time_steps,
        dropout,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.dropout = dropout

        # Initialize LSTM unit
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,
        )

        # The linear layer that maps from hidden state space to tag space
        self.hidden2out = nn.Linear(hidden_dim, target_size)
        self.hidden = self.init_hidden()

        self.drop = nn.Dropout(dropout)

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size , hidden_dim)
        return (
            torch.zeros(self.num_layers, self.time_steps, self.hidden_dim),
            torch.zeros(self.num_layers, self.time_steps, self.hidden_dim),
        )

    def forward(self, input_seq):
        lstm_out, self.hidden = self.lstm(input_seq, self.hidden)

        drop_out = self.drop(lstm_out)

        pred = self.hidden2out(drop_out)

        return pred
