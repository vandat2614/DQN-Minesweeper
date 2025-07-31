import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Type

ACTIVATION_MAP = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'LeakyReLU': nn.LeakyReLU,
    'Sigmoid': nn.Sigmoid,
    'Softmax': lambda: nn.Softmax(dim=-1),
    None: nn.Identity
}

POOLING_MAP = {
    "MaxPool2d": lambda: nn.MaxPool2d(kernel_size=2, stride=2),
    "AvgPool2d": lambda: nn.AvgPool2d(kernel_size=2, stride=2),
    "AdaptiveAvgPool2d": lambda: nn.AdaptiveAvgPool2d((1, 1)),
    "AdaptiveMaxPool2d": lambda: nn.AdaptiveMaxPool2d((1, 1)),
    None: nn.Identity,
}


class ConvNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (C, H, W)
        kernel_sizes: List[int],
        strides: List[int],
        paddings: List[int],
        out_channels: List[int],
        conv_activations: List[Type[nn.Module]],
        poolings: List[Optional[Type[nn.Module]]],
        hidden_sizes: List[int],
        hidden_activations: List[Type[nn.Module]],
        output_size: int,
        output_activation: Optional[Type[nn.Module]] = None,
        init_method: str = "xavier_uniform"
    ):
        super(ConvNeuralNetwork, self).__init__()

        conv_layers = []
        self.layers_to_init = []

        for i in range(len(kernel_sizes)):
            in_channels = input_shape[0] if i == 0 else out_channels[i - 1]
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                bias=True
            )
            activation = conv_activations[i]()
            pooling = poolings[i]()
            conv_layers.extend([conv, activation, pooling])
            self.layers_to_init.append(conv)

        self.conv_layers = nn.Sequential(*conv_layers)
        flatten_size = self._caculate_features_dim(input_shape)

        fc_layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            in_features = flatten_size if i == 0 else hidden_sizes[i - 1]
            linear = nn.Linear(in_features, hidden_size, bias=True)
            fc_layers.extend([linear, hidden_activations[i]()])
            self.layers_to_init.append(linear)
        
        output_layer = nn.Linear(hidden_sizes[-1], output_size)
        fc_layers.extend([output_layer, output_activation()])
        self.layers_to_init.append(output_layer)

        self.fc_layers = nn.Sequential(*fc_layers)

        self._initialize_weights(init_method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x

    def _caculate_features_dim(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_output = self.conv_layers(dummy_input)
            flattened_size = conv_output.view(1, -1).shape[1]
        return flattened_size

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _initialize_weights(self, method: str):
        for layer in self.layers_to_init:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                if method == "xavier_uniform":
                    nn.init.xavier_uniform_(layer.weight)
                elif method == "xavier_normal":
                    nn.init.xavier_normal_(layer.weight)
                elif method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                elif method == "kaiming_normal":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                elif method == "orthogonal":
                    nn.init.orthogonal_(layer.weight)
                elif method == "normal":
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                elif method == "uniform":
                    nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
                else:
                    raise ValueError(f"Unsupported init method: {method}")

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    @classmethod
    def from_config(cls, config: dict, input_shape : tuple, output_size : int):
        return cls(
            input_shape=input_shape,
            kernel_sizes=config["kernel_sizes"],
            strides=config["strides"],
            paddings=config["paddings"],
            out_channels=config["out_channels"],
            conv_activations=[ACTIVATION_MAP[act] for act in config["conv_activations"]],
            poolings=[POOLING_MAP[pool] for pool in config["poolings"]],
            hidden_sizes=config["hidden_sizes"],
            hidden_activations=[ACTIVATION_MAP[act] for act in config["hidden_activations"]],
            output_size=output_size,
            output_activation=ACTIVATION_MAP[config["output_activation"]],
            init_method=config.get("init_method", "xavier_uniform")
        )

    def load(self, path: str):
        device = next(self.parameters()).device
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)