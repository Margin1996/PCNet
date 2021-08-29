import torch
from torch import nn
from utils import DRB,PNB,ChannelSELayer



class CloudDetector(nn.Module):
    def __init__(self):
        super(CloudDetector,self).__init__()
        self.in_conv_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=False))
        self.in_conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=False))


        self.PNB1 = PNB(in_num=64,filter_num=32)
        self.DRB1 = DRB(channels=64)
        self.PNB2 = PNB(in_num=64,filter_num=32)
        self.DRB2 = DRB(channels=64)
        self.ChannelSE = ChannelSELayer(num_channels=64*3)

        self.out_con1 = nn.Sequential(
            nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=False))

        self.out_con2 = nn.Conv2d(64,1,1,1)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def forward(self, input):
        x = self.in_conv_1(input)
        f1 = self.in_conv_2(x)
        f2 = self.PNB1(self.DRB1(f1))
        f3 = self.PNB2(self.DRB2(f2))

        fusion = torch.cat((f1, f2, f3), 1)
        fusion = self.ChannelSE(fusion)
        result = self.out_con1(fusion)
        result = self.out_con2(result)
        
        return self.sigmoid(result)
