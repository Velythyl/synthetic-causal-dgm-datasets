# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

import nflows.transforms
import nflows.utils
import torch
from nflows.nn.nets import ResidualNet


def make_scalar_transform(
    n_features,
    layers=3,
    hidden=10,
    transform_blocks=1,
    sigmoid=False,
    transform="affine",
    conditional_features=0,
    bins=10,
    tail_bound=10.0,
):
    """Utility function that constructs an invertible transformation for unstructured data"""
    def transform_net_factory_fn(in_features, out_features):
        # noinspection PyUnresolvedReferences
        return ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden,
            context_features=conditional_features,
            num_blocks=transform_blocks,
            activation=torch.nn.functional.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
        )

    transforms = []
    for i in range(layers):
        transforms.append(nflows.transforms.RandomPermutation(features=n_features))
        if transform == "affine":
            transforms.append(
                nflows.transforms.AffineCouplingTransform(
                    mask=nflows.utils.create_alternating_binary_mask(n_features, even=(i % 2 == 0)),
                    transform_net_create_fn=transform_net_factory_fn,
                )
            )
        elif transform == "piecewise_linear":
            transforms.append(
                nflows.transforms.PiecewiseLinearCouplingTransform(
                    mask=nflows.utils.create_alternating_binary_mask(n_features, even=(i % 2 == 0)),
                    transform_net_create_fn=transform_net_factory_fn,
                    tail_bound=tail_bound,
                    num_bins=bins,
                    tails="linear",
                )
            )
        else:
            raise ValueError(transform)
    transforms.append(nflows.transforms.RandomPermutation(features=n_features))
    if sigmoid:
        transforms.append(nflows.transforms.Sigmoid())

    return nflows.transforms.CompositeTransform(transforms)

