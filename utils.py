import torch
import torch.nn.functional as F
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out + residual
        return out

class DRB(nn.Module):
    def __init__(self,channels):
        super(DRB, self).__init__()
        self.con1 = BasicBlock(channels,dilation=1)
        self.con2 = BasicBlock(channels,dilation=2)
        self.con3 = BasicBlock(channels,dilation=4)
        self.con4 = BasicBlock(channels,dilation=2)
        self.con5 = BasicBlock(channels,dilation=1)
    def forward(self, input):
        x1 = self.con1(input)
        x2 = self.con2(x1)
        x3 = self.con3(x2)
        x4 = self.con4(x3)
        x5 = self.con5(x4)
        return x5
class PNB(nn.Module):
    def __init__(self,in_num,filter_num):
        super(PNB, self).__init__()
        self.filter_num = filter_num
        self.stride = [4,8,16]   #8 16 32
        self.conv1 = nn.Conv2d(in_num, in_num, 1, 1)
        self.conv2 = nn.Conv2d(in_num, in_num, self.stride[0], self.stride[0])
        self.conv3 = nn.Conv2d(in_num, filter_num, self.stride[0], self.stride[0])

        self.conv4 = nn.Conv2d(in_num, in_num, self.stride[1], self.stride[1])
        self.conv5 = nn.Conv2d(in_num, filter_num, self.stride[1], self.stride[1])

        self.conv6 = nn.Conv2d(in_num, in_num, self.stride[2], self.stride[2])
        self.conv7 = nn.Conv2d(in_num, filter_num, self.stride[2], self.stride[2])

        self.out_conv = nn.Conv2d(3 * filter_num, in_num, 1, 1)
        self.alpha = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, input):
        batch_size,in_channel,h,w = input.size()
        theta = self.conv1(input).view(batch_size,in_channel,-1).permute(0,2,1) #B*WH*C

        x1 = self.conv2(input).view(batch_size,in_channel,-1) #B*C*W1H1
        attention_1 = self.softmax(torch.bmm(theta,x1)) #B*WH*W1H1
        g1 = self.conv3(input).view(batch_size,self.filter_num,-1).permute(0,2,1) #B*W1H1*N
        E1 = torch.bmm(attention_1,g1).view(batch_size,h,w,-1).permute(0,3,1,2) #B*N*H*W

        x2 = self.conv4(input).view(batch_size,in_channel,-1)
        attention_2 = self.softmax(torch.bmm(theta,x2)) #B*WH*W2H2
        g2 = self.conv5(input).view(batch_size,self.filter_num,-1).permute(0,2,1)#B*W2H2*N
        E2 = torch.bmm(attention_2,g2).view(batch_size,h,w,-1).permute(0,3,1,2) #B*N*H*W

        x3 = self.conv6(input).view(batch_size, in_channel, -1)
        attention_3 = self.softmax(torch.bmm(theta, x3))  # B*WH*W3H3
        g3 = self.conv7(input).view(batch_size, self.filter_num, -1).permute(0, 2, 1)  # B*W2H2*N
        E3 = torch.bmm(attention_3, g3).view(batch_size, h, w, -1).permute(0, 3, 1, 2)  # B*N*H*W

        fusion = torch.cat((E1,E2,E3),dim = 1)
        out = self.alpha*self.out_conv(fusion)+input

        return out
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """
    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

