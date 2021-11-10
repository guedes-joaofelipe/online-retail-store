import torch.nn as nn

class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        num_layers: int
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.BatchNorm1d(input_dim))
        for n in range(num_layers):
            if n < (num_layers - 1):
                self.layers.append(nn.Linear(input_dim, hidden_dim))
                nn.init.kaiming_normal_(self.layers[-1].weight, mode="fan_in")
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
        self.layers.append(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x