# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np


class Swish(torch.nn.Module):
    """
    Swish module
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class NeuralNet_General(torch.nn.Module): # input-hidden-output

    def __init__(self, num_features, activation='sigmoid', add_batch_norm=False, momentum=0.9):
        super(NeuralNet_General, self).__init__()
        self.Layers = torch.nn.ModuleList([])
        self.BatchNorms = torch.nn.ModuleList([])
        self.add_batch_norm = add_batch_norm
        num_layers = len(num_features)
        for layer_idx in range(num_layers-1):
            input_size = num_features[layer_idx]
            output_size = num_features[layer_idx+1]
            layer = torch.nn.Linear(input_size, output_size)
            self.Layers.append(layer)

            if add_batch_norm and layer_idx < num_layers-2:
                # The convention for momentum in BatchNorm1d is different than momentum in the optimizer, i.e.,
                # 1.0 - optimizer momentum
                bn = torch.nn.BatchNorm1d(output_size, momentum=1.0-momentum)
                self.BatchNorms.append(bn)

        self.activation_func = None
        activation = activation.lower()
        if activation == 'sigmoid':
            self.activation_func = torch.nn.Sigmoid()
        elif activation == 'relu':
            self.activation_func = torch.nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.activation_func = torch.nn.LeakyReLU(inplace=True)
        elif activation == 'swish':
            self.activation_func = Swish()
        else:
            raise NotImplementedError('activation: {} is not implemented'.format(activation))

    def forward(self, features_0):

        num_layers = len(self.Layers)
        x = features_0
        for layer_idx in range(num_layers-1):
            x = self.Layers[layer_idx](x)
            x = self.activation_func(x)
            if self.add_batch_norm:
                x = self.BatchNorms[layer_idx](x)

        x = self.Layers[-1](x)
        return x


class AENet_General(torch.nn.Module):

    def __init__(self, num_features,
                 activation='sigmoid',
                 depth_to_feature_representation=3,
                 width_of_feature_representation=6,
                 momentum=0.9,
                 add_batch_norm=False):

        super(AENet_General, self).__init__()
        reduction_amount = float(num_features - width_of_feature_representation) / depth_to_feature_representation
        self.Layers = torch.nn.ModuleList([])
        self.BatchNorms = torch.nn.ModuleList([])
        self.add_batch_norm = add_batch_norm
        input_size = num_features
        input_sizes = []
        output_sizes = []

        # Encoder
        for layer_idx in range(depth_to_feature_representation):
            if layer_idx < depth_to_feature_representation-1:
                output_size = int(np.round_(input_size - reduction_amount))
            else:
                output_size = width_of_feature_representation
            input_sizes.append(input_size)
            output_sizes.append(output_size)
            layer = torch.nn.Linear(input_size, output_size)
            self.Layers.append(layer)

            if add_batch_norm:
                # The convention for momentum in BatchNorm1d is different than momentum in the optimizer, i.e.,
                # 1.0 - optimizer momentum
                bn = torch.nn.BatchNorm1d(output_size, momentum=1.0-momentum)
                self.BatchNorms.append(bn)

            input_size = output_size

        # Decoder
        for layer_idx in range(len(input_sizes)-1, -1, -1):
            input_size = output_sizes[layer_idx]  # decoder is mirrored
            output_size = input_sizes[layer_idx]
            layer = torch.nn.Linear(input_size, output_size)
            self.Layers.append(layer)
            if add_batch_norm and layer_idx > 0:
                # The convention for momentum in BatchNorm1d is different than momentum in the optimizer, i.e.,
                # 1.0 - optimizer momentum
                bn = torch.nn.BatchNorm1d(output_size, momentum=1.0-momentum)
                self.BatchNorms.append(bn)

        self.activation_func = None
        activation = activation.lower()
        if activation == 'sigmoid':
            self.activation_func = torch.nn.Sigmoid()
        elif activation == 'relu':
            self.activation_func = torch.nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.activation_func = torch.nn.LeakyReLU(inplace=True)
        elif activation == 'swish':
            self.activation_func = Swish()
        else:
            raise NotImplementedError('activation: {} is not implemented'.format(activation))

    def forward(self, features_0):

        num_layers = len(self.Layers)
        x = features_0
        for layer_idx in range(num_layers-1):
            x = self.Layers[layer_idx](x)
            x = self.activation_func(x)
            if self.add_batch_norm:
                x = self.BatchNorms[layer_idx](x)
        x = self.Layers[-1](x)
        return x
