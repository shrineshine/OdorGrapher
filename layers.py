import torch
import torch.nn as nn
from typing import List, Optional, Callable, Any

class CustomPositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        d_input: int = 1024,
        d_hidden_list: List = [1024],
        d_output: int = 1024,
        activation: str = 'leakyrelu',
        dropout_p: float = 0.0,
        dropout_at_input_no_act: bool = False,
        batch_norm: bool = True,
    ):
        super().__init__()

        self.dropout_at_input_no_act = dropout_at_input_no_act
        self.batch_norm = batch_norm

        # activation mapping
        activations = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(0.1),
            'prelu': nn.PReLU(),
            'tanh': nn.Tanh(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'linear': nn.Identity(),
        }
        self.activation = activations[activation]

        d_output = d_output if d_output != 0 else d_input
        hidden_dims = d_hidden_list

        layer_dims = [d_input] + hidden_dims + [d_output]
        self.linears = nn.ModuleList(
            nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)
        )

        self.dropout_p = nn.ModuleList(nn.Dropout(dropout_p) for _ in range(len(self.linears)))

        if batch_norm and hidden_dims:
            self.batchnorms = nn.ModuleList(nn.BatchNorm1d(h) for h in hidden_dims)

    def forward(self, x: torch.Tensor) -> List[Optional[torch.Tensor]]:
        n_layers = len(self.linears)

        if n_layers == 1:
            if self.dropout_at_input_no_act:
                x = self.dropout_p[0](x)
                return [None, self.linears[0](x)]
            else:
                x = self.activation(self.linears[0](x))
                x = self.dropout_p[0](x)
                return [None, x]

        if self.dropout_at_input_no_act:
            x = self.dropout_p[-1](x)

        for i in range(n_layers - 2):
            h = self.linears[i](x)
            if self.batch_norm:
                h = self.batchnorms[i](h)
            x = self.dropout_p[i](self.activation(h))

        embeddings = self.linears[-2](x)

        if self.batch_norm:
            x = self.batchnorms[-1](embeddings)
            x = self.activation(x)
        else:
            x = self.activation(embeddings)

        x = self.dropout_p[-2](x)

        output = self.linears[-1](x)
        return [embeddings, output]