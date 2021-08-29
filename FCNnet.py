# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 18:36:22 2020

@author: gaosong
"""

import os
import numpy as np
import torch
import torchvision
from torch.autograd import variable
from torchvision import datasets, transforms
import torch.nn as nn
import cv2
import torch.utils.data as Data

#网络搭建
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.net1_conv1 = nn.Conv2d(3,64,3, padding = 1)
        self.net1_conv2 = nn.Conv2d(64,64,3, padding = 1)
        self.net1_BN1 = nn.BatchNorm2d(64)
        self.net1_conv3 = nn.Conv2d(128,64,3, padding = 1)
        self.net1_conv4 = nn.Conv2d(64,32,3, padding = 1)
        self.net1_conv5 = nn.Conv2d(32,1,1)
        
        self.net2_max_pool = nn.MaxPool2d(2)
        self.net2_conv1 = nn.Conv2d(64,128,3, padding = 1)
        self.net2_conv2 = nn.Conv2d(128,128,3, padding = 1)
        self.net2_BN1 = nn.BatchNorm2d(128)
        self.net2_conv3 = nn.Conv2d(256,128,3, padding = 1)
        self.net2_conv4 = nn.Conv2d(128,64,3, padding = 1)
        self.net2_conv5 = nn.Conv2d(64,32,3, padding = 1)
        self.net2_conv6 = nn.Conv2d(32,1,1)
        self.net2_convtrans = nn.ConvTranspose2d(64, 64, 2, 2)
        
        self.net3_max_pool = nn.MaxPool2d(2)
        self.net3_conv1 = nn.Conv2d(128,256,3, padding = 1)
        self.net3_conv2 = nn.Conv2d(256,256,3, padding = 1)
        self.net3_BN1 = nn.BatchNorm2d(256)
        self.net3_conv3 = nn.Conv2d(256,256,3, padding = 1)
        self.net3_BN2 = nn.BatchNorm2d(256)
        self.net3_conv4 = nn.Conv2d(512,256,3, padding = 1)
        self.net3_conv5 = nn.Conv2d(256,128,3, padding = 1)
        self.net3_conv6 = nn.Conv2d(128,32,3, padding = 1)
        self.net3_conv7 = nn.Conv2d(32,1,1)
        self.net3_convtrans = nn.ConvTranspose2d(128, 128, 2, 2)
        
        self.net4_max_pool = nn.MaxPool2d(2)
        self.net4_conv1 = nn.Conv2d(256,512,3, padding = 1)
        self.net4_conv2 = nn.Conv2d(512,512,3, padding = 1)
        self.net4_BN1 = nn.BatchNorm2d(512)
        self.net4_conv3 = nn.Conv2d(512,512,3, padding = 1)
        self.net4_BN2 = nn.BatchNorm2d(512)
        self.net4_conv4 = nn.Conv2d(1024,512,3, padding = 1)
        self.net4_conv5 = nn.Conv2d(512,256,3, padding = 1)
        self.net4_conv6 = nn.Conv2d(256,32,3, padding = 1)
        self.net4_conv7 = nn.Conv2d(32,1,1)
        self.net4_convtrans = nn.ConvTranspose2d(256, 256, 2, 2)
        
        self.net5_max_pool = nn.MaxPool2d(2)
        self.net5_conv1 = nn.Conv2d(512,512,3, padding = 1)
        self.net5_conv2 = nn.Conv2d(512,512,3, padding = 1)
        self.net5_BN1 = nn.BatchNorm2d(512)
        self.net5_conv3 = nn.Conv2d(512,512,3, padding = 1)
        self.net5_BN2 = nn.BatchNorm2d(512)
        self.net5_convtrans = nn.ConvTranspose2d(512, 512, 2, 2)
        
        self.upsample1 = nn.Upsample(512)
        self.upsample2 = nn.Upsample(512)
        self.upsample3 = nn.Upsample(512)
        
    def forward(self, x):
        
        x1 = self.net1_conv1(x)
        x1 = self.relu(x1)
        x1 = self.net1_conv2(x1)
        x1 = self.net1_BN1(x1)
        x1 = self.relu(x1)
        
        
        x2 = self.net2_max_pool(x1)
        x2 = self.net2_conv1(x2)
        x2 = self.relu(x2)
        x2 = self.net2_conv2(x2)
        x2 = self.net2_BN1(x2)
        x2 = self.relu(x2)
        
        x3 = self.net3_max_pool(x2)
        x3 = self.net3_conv1(x3)
        x3 = self.relu(x3)
        x3 = self.net3_conv2(x3)
        x3 = self.net3_BN1(x3)
        x3 = self.relu(x3)
        x3 = self.net3_conv3(x3)
        x3 = self.net3_BN2(x3)
        x3 = self.relu(x3)
        
        x4 = self.net4_max_pool(x3)
        x4 = self.net4_conv1(x4)
        x4 = self.relu(x4)
        x4 = self.net4_conv2(x4)
        x4 = self.net4_BN1(x4)
        x4 = self.relu(x4)
        x4 = self.net4_conv3(x4)
        x4 = self.net4_BN2(x4)
        x4 = self.relu(x4)
        
        x5 = self.net5_max_pool(x4)
        x5 = self.net5_conv1(x5)
        x5 = self.relu(x5)
        x5 = self.net5_conv2(x5)
        x5 = self.net5_BN1(x5)
        x5 = self.relu(x5)
        x5 = self.net5_conv3(x5)
        x5 = self.net5_BN2(x5)
        x5 = self.relu(x5)
        
        x5 = self.net5_convtrans(x5) 
        x4 = torch.cat((x4,x5),1)
        x4 = self.net4_conv4(x4)
        x4 = self.relu(x4)
        x4 = self.net4_conv5(x4)
        x4 = self.relu(x4)
        
        # out4 = self.net4_conv6(x4)
        # out4 = self.relu(out4)
        # out4 = self.net4_conv7(out4)
        # out4 = self.upsample3(out4)
        # out4 = self.sigmoid(out4)
        # out4 = torch.clamp(out4, 1e-4, 1-1e-4)
        
        
        x4 = self.net4_convtrans(x4)
        x3 = torch.cat((x3,x4),1)
        x3 = self.net3_conv4(x3)
        x3 = self.relu(x3)
        x3 = self.net3_conv5(x3)
        x3 = self.relu(x3)
        
        # out3 = self.net3_conv6(x3)
        # out3 = self.relu(out3)
        # out3 = self.net3_conv7(out3)
        # out3 = self.upsample2(out3)
        # out3 = self.sigmoid(out3)
        # out3 = torch.clamp(out3, 1e-4, 1-1e-4)
        
        
        x3 = self.net3_convtrans(x3)
        x2 = torch.cat((x2,x3),1)
        x2 = self.net2_conv3(x2)
        x2 = self.relu(x2)
        x2 = self.net2_conv4(x2)
        x2 = self.relu(x2)

        
        # out2 = self.net2_conv5(x2)
        # out2 = self.relu(out2)
        # out2 = self.net2_conv6(out2)
        # out2 = self.upsample1(out2)
        # out2 = self.sigmoid(out2)
        # out2 = torch.clamp(out2, 1e-4, 1-1e-4)
        
        
        x2 = self.net2_convtrans(x2)
        x1 = torch.cat((x1,x2),1)
        x1 = self.net1_conv3(x1)
        x1 = self.relu(x1)
        x1 = self.net1_conv4(x1)
        x1 = self.relu(x1)
        x1 = self.net1_conv5(x1)
        out1 = self.sigmoid(x1)
        out1 = torch.clamp(out1, 1e-4, 1-1e-4)
        
        return out1

# def cross_entropy(prediction, label):
#     return -torch.mean(label * torch.log(prediction) + (1 - label) * torch.log(1 - prediction))


        
        
        
        
