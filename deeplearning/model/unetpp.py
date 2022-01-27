import torch
import torch.nn as nn
import torch.nn.functional as F

import os
 
#Reference: https://github.com/amirfaraji/LowDoseCTPytorch

class UnetConv2dBlock(nn.Module):

    def __init__(self, in_channel: int, mid_channel: int, out_channel: int, kernel_size: list = [3,3], stride_size: list = [1,1], activation: str = 'relu'):
        super().__init__()

        self.conv2dblock = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=kernel_size[0], stride=stride_size[0], padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=kernel_size[1], stride=stride_size[1], padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv2dblock(x)

class DownSample(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxpooling = nn.MaxPool2d(2)

    def forward(self, x):
        return self.maxpooling(x)

class UpSample(nn.Module):

    def __init__(self, in_channel=None, bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channel , in_channel // 2, kernel_size=2, stride=2)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

class UnetPlusPlus(nn.Module):
    
    def __init__(self, model_id, project_dir, in_channel = 3, num_classes = 4):
        super(UnetPlusPlus, self).__init__()
        
        self.num_classes = num_classes

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        num_filter = [32, 64, 128, 256, 512]

        self.down = DownSample()
        self.up = UpSample()

        self.conv_block_0_0 = UnetConv2dBlock(in_channel, num_filter[0], num_filter[0])
        self.conv_block_1_0 = UnetConv2dBlock(num_filter[0], num_filter[1], num_filter[1])
        self.conv_block_2_0 = UnetConv2dBlock(num_filter[1], num_filter[2], num_filter[2])
        self.conv_block_3_0 = UnetConv2dBlock(num_filter[2], num_filter[3], num_filter[3])
        
        self.conv_block_0_1 = UnetConv2dBlock(num_filter[0]+num_filter[1], num_filter[0], num_filter[0])
        self.conv_block_1_1 = UnetConv2dBlock(num_filter[1]+num_filter[2], num_filter[1], num_filter[1])
        self.conv_block_2_1 = UnetConv2dBlock(num_filter[2]+num_filter[3], num_filter[2], num_filter[2])

        self.conv_block_0_2 = UnetConv2dBlock(2*num_filter[0]+num_filter[1], num_filter[0], num_filter[0])
        self.conv_block_1_2 = UnetConv2dBlock(2*num_filter[1]+num_filter[2], num_filter[1], num_filter[1])

        self.conv_block_0_3 = UnetConv2dBlock(3*num_filter[0]+num_filter[1], num_filter[0], num_filter[0])

        self.final = nn.Conv2d(num_filter[0], self.num_classes, kernel_size=1)

    def forward(self, x):

        x0_0 = self.conv_block_0_0(x)

        x1_0 = self.conv_block_1_0(self.down(x0_0))
        x0_1 = self.conv_block_0_1(self.up(x1_0, x0_0))
        
        x2_0 = self.conv_block_2_0(self.down(x1_0))
        x1_1 = self.conv_block_1_1(self.up(x2_0, x1_0))
        x0_2 = self.conv_block_0_2(self.up(x1_1, torch.cat([x0_0, x0_1], 1)))

        x3_0 = self.conv_block_3_0(self.down(x2_0))
        x2_1 = self.conv_block_2_1(self.up(x3_0, x2_0))
        x1_2 = self.conv_block_1_2(self.up(x2_1, torch.cat([x1_0, x1_1],1)))
        x0_3 = self.conv_block_0_3(self.up(x1_2, torch.cat([x0_0, x0_1, x0_2], 1)))

        out = self.final(x0_3)
        return out

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)