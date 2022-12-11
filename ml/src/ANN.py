from torch import nn


class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_size, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            hidden_dim = int(hidden_dim / 2) if i > 0 else hidden_dim
            if hidden_dim < 2:
                hidden_dim = 2
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        self.layers.append(nn.Linear(hidden_dim, target_size))
        self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
