# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

"""
General-purpose neural networks
"""

from torch import nn


def get_activation(key):
    """Utility function that returns an activation function given its name"""

    if key == "relu":
        return nn.ReLU()
    elif key == "leaky_relu":
        return nn.LeakyReLU()
    elif key == "softmax":
        return nn.Softmax(dim=1)
    else:
        raise ValueError(f"Unknown activation {key}")


def make_mlp(features, activation="relu", final_activation=None, initial_activation=None):
    """Utility function that constructs a simple MLP from specs"""

    if len(features) >= 2:
        layers = []

        if initial_activation is not None:
            layers.append(get_activation(initial_activation))

        for in_, out in zip(features[:-2], features[1:-1]):
            layers.append(nn.Linear(in_, out))
            layers.append(get_activation(activation))

        layers.append(nn.Linear(features[-2], features[-1]))
        if final_activation is not None:
            layers.append(get_activation(final_activation))

        net = nn.Sequential(*layers)

    else:
        net = nn.Identity()

    return net
