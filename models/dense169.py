import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class _DenseLayer(nn.Module):
    def __init__(self, n_input_features, bn_sz, growth_rate):
        super(_DenseLayer, self).__init__()
        dense_layer = nn.Sequential()
        dense_layer.add_module("batch_norm1", nn.BatchNorm2d(n_input_features) )
        dense_layer.add_module("relu1", nn.ReLU(inplace=True))
        dense_layer.add_module("conv1", nn.Conv2d(n_input_features, bn_sz * growth_rate,
                                            1, stride=1, bias=False))
        # 1X1 convolutions maintain size, so no padding
        dense_layer.add_module("batch_norm2", nn.BatchNorm2d(bn_sz * growth_rate))
        dense_layer.add_module("relu2", nn.ReLU(inplace=True))
        dense_layer.add_module("conv2", nn.Conv2d(bn_sz*growth_rate, growth_rate, 3,
                                            stride=1, padding=1, bias=False))
        # 3x3 convs decrease size, so add padding =1 to maintain size

    def forward(self, *input):

        X = self.dense_layer(input)
        #bottleneck_out = self.named_children()
        return X

class _DenseBlock(nn.Module):
    def __init__(self, n_layers, n_input_features, bn_sz, growth_rate):
        super(_DenseBlock, self).__init__()
        #self.layer = []
        for i in range(n_layers):
            layer = _DenseLayer(
                n_input_features=(i*growth_rate) + n_input_features,
                bn_sz = bn_sz, 
                growth_rate=growth_rate)

            self.add_module('layer(%d+1)'%(i+1), layer)

    def forward(self, input):
        all_features = [input]
        for name, layer in self.named_children():
            out_features = layer(*input)
            all_features.append(out_features)
        return torch.cat(all_features, 1)


        
class _TransitionLayer(nn.Module):
    def __init__(self, n_in_features, n_out_features):
        super(_TransitionLayer, self).__init__()
        self.transition = nn.Sequential()
        self.bn1 = transition.add_module('norm1', nn.BatchNorm2d(n_in_features))
        self.relu = transition.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1 = transition.add_module('conv1', nn.Conv2d(n_in_features, n_out_features, 1,
                                           stride=1, bias=False))
        self.pool1 = transition.add_module('avg_pool', nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, input):
        X = self.bn1(input)
        X = self.conv1(self.relu(X))
        X = self.pool1(X)
        return X



class Dense169(nn.Module):
    """dense_blk_config is a set of 4 numbers - each number number of blocks in
    a dense layer - for densenet 169 it is 6, 12, 32, 32"""
    def __init__(self, n_layers, growth_rt, n_init_features=64,
                 bn_sz =4, dense_blk_config=(6, 12, 32, 32)):
        super(Dense169, self).__init__()
        
        #inital convolutional layer just after the input
        #Changed from the original 7x7, stride =2 conv, to suit our
        #rectangular images
        self.network = nn.Sequential(
            nn.Conv2d(3, n_init_features, kernel_size=(7,7), stride=(2, 1),
            padding=(1, 3), bias=True), #output = 64w x 98h
            nn.BatchNorm2d(n_init_features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=2, padding=1)
        )

        #Start with initial num of layers, and add denseblock n times, and 
        #one 'transition layer' where n = num of dense layers in one dense blk
        #n = 6, transition, then n = 12, followed by transition, n=32 ....

        n_features = n_init_features
        for i, n_layers in enumerate(dense_blk_config):
            block = _DenseBlock(
                n_layers=n_layers,
                n_input_features=n_features,
                bn_sz=bn_sz,
                growth_rt=growth_rt, 
            )
            self.network.add_module('dense_block%d'%(i+1), block)
            n_features = (growth_rt * n_layers) + n_features
            if i != len(dense_blk_config) - 1:
                self.network.add_module('transition%d'%(i+1), 
                _TransitionLayer(n_in_featues=n_features,
                                 n_out_features=n_features//2)
                )
                n_features = n_features//2


        #n_features = 1664 here, img size = 8 x 12
        self.network.add_module('norm_final', nn.BatchNorm2d(n_features))
        #self.network.add_module('relu_final', nn.ReLU(inplace=True))
        #self.network.add_module('adapool_final', nn.AdaptiveAvgPool2d((2, 3)) ) 
        self.linear_layer = nn.Linear(n_features*2*3, 50*62)

        
    def forward(self, input):
        X = self.network(input)
        X = F.relu(X, inplace=True)
        X = F.adaptive_avg_pool2d(X, (2, 3))
        X = self.linear_layer(X)
        return X
    
#d169 = Dense169(3, 32, 3,)