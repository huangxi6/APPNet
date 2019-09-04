import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools
import pdb

import sys, os

from libs import InPlaceABN, InPlaceABNSync
import torch.autograd as autograd


BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)

        return out


class Feature_extraction_module(nn.Module):
    def __init__(self, features, inner_features=256, out_features=256, dilations=(3, 8, 12)):
        super(Feature_extraction_module, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv2 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   InPlaceABNSync(inner_features),
                                   nn.Conv2d(inner_features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   InPlaceABNSync(inner_features),
                                   nn.Conv2d(inner_features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   InPlaceABNSync(inner_features),
                                   nn.Conv2d(inner_features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(inner_features))
        
    def forward(self, x):

        _, _, h, w = x.size()

        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)

        out = feat1 + feat2 + feat3 + feat4

        return out

class Global_attention_low(nn.Module):
    def __init__(self):
        super(Global_attention_low, self).__init__()
        self.extractor_a = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256))
        self.extractor_b = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256))

        self.conv1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(128))
        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(128))
        self.softmax = nn.Softmax(dim=-1)

        self.last_layer = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=True),
            InPlaceABNSync(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, 2, kernel_size=1, padding=0, dilation=1, bias=True))

    def forward(self, x_a, x_b):
        _, _, h, w = x_a.size()
        feature_a = self.extractor_a(x_a)
        feature_b = F.interpolate(self.extractor_b(x_b), size=(h, w), mode='bilinear',align_corners=True)

        x = torch.cat((feature_a, feature_b), dim=1)

        feature_a = self.conv1(feature_a)
        feature_b = self.conv2(feature_b)
        feature_a = feature_a.mean(1, keepdim=True)
        feature_b = feature_b.mean(1, keepdim=True)
        global_weight = feature_a + feature_b
        global_weight = self.softmax(global_weight)
        x = x * global_weight

        seg = self.last_layer(x)

        return x,seg

class Global_attention_high(nn.Module):
    def __init__(self):
        super(Global_attention_high, self).__init__()
        self.extractor_a = Feature_extraction_module(1024)
        self.extractor_b = Feature_extraction_module(2048)

        self.conv1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(128))
        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(128))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_a, x_b):
        _, _, h, w = x_a.size()
        feature_a = self.extractor_a(x_a)
        feature_b = self.extractor_b(x_b)

        x = torch.cat((feature_a, feature_b), dim=1)

        feature_a = self.conv1(feature_a)
        feature_b = self.conv2(feature_b)
        feature_a = feature_a.mean(1, keepdim=True)
        feature_b = feature_b.mean(1, keepdim=True)
        global_weight = feature_a + feature_b
        global_weight = self.softmax(global_weight)
        x = x * global_weight

        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1,1,1))
        
        ##progressive
        self.global_attention_low = Global_attention_low()
        self.global_attention_high = Global_attention_high()

        self.last_layer = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=True),
            InPlaceABNSync(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, padding=0, dilation=1, bias=True))


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        _, _, h, w = x2.size()

        #########Progressive
        x_low, seg1 = self.global_attention_low(x2, x3)
        x_high = F.interpolate(self.global_attention_high(x4, x5), size=(h, w), mode='bilinear',align_corners=True)
        x_merge = x_low + x_high
        seg = self.last_layer(x_merge)

        return [[seg], [seg1]]


def Res_Deeplab(num_classes=21):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model



