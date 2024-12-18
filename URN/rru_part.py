import baal
import torch
import torch.nn as nn
import torch.nn.functional as F

# ~~~~~~~~~~ U-Net ~~~~~~~~~~
import torchsnooper


class U_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(U_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = U_double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class U_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(U_down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            U_double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class U_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(U_up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = U_double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffY, 0,
                        diffX, 0))
        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)
        return x


# ~~~~~~~~~~ RU-Net ~~~~~~~~~~

class RU_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RU_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.conv(x)
        return x


class RU_double_conv_MCD(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.5):
        super(RU_double_conv_MCD, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            baal.bayesian.dropout.Dropout(p=dropout_rate),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.conv(x)
        return x


class RU_first_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RU_first_down, self).__init__()
        self.conv = RU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))

        return r1


class RU_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RU_down, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = RU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.maxpool(x)
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))

        return r1


class RU_up(nn.Module):
    def __init__(self, out_ch, in_ch, in_ch_skip=0, bilinear=False, with_skip=True):
        super(RU_up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        #  nn.Upsample hasn't weights to learn, but nn.ConvTransposed2d has weights to learn.
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            if in_ch_skip == 0 and with_skip:
                self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            else:
                self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.conv = RU_double_conv(in_ch + in_ch_skip, out_ch)
        self.relu = nn.ReLU(inplace=True)
        group_num = 32
        if out_ch % 32 == 0 and out_ch >= 32:
            if out_ch % 24 == 0:
                group_num = 24
        elif out_ch % 16 == 0 and out_ch >= 16:
            if out_ch % 16 == 0:
                group_num = 16
        # print(out_ch, group_num)
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch + in_ch_skip, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, out_ch))
        self.with_skip = with_skip

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.with_skip:
            diff_x = x2.size()[-2] - x1.size()[-2]
            diff_y = x2.size()[-1] - x1.size()[-1]

            x1 = F.pad(x1, (diff_y, 0, diff_x, 0))
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(self.res_conv(x) + ft1)

        return r1


class RU_up_MCD(nn.Module):
    def __init__(self, out_ch, in_ch, in_ch_skip=0, bilinear=False, with_skip=True, dropout_rate=0.5):
        super(RU_up_MCD, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        #  nn.Upsample hasn't weights to learn, but nn.ConvTransposed2d has weights to learn.
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            if in_ch_skip == 0 and with_skip:
                self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            else:
                self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.conv = RU_double_conv_MCD(in_ch + in_ch_skip, out_ch, dropout_rate=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        group_num = 32
        if out_ch % 32 == 0 and out_ch >= 32:
            if out_ch % 24 == 0:
                group_num = 24
        elif out_ch % 16 == 0 and out_ch >= 16:
            if out_ch % 16 == 0:
                group_num = 16
        # print(out_ch, group_num)
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch + in_ch_skip, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, out_ch))
        self.with_skip = with_skip

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.with_skip:
            diff_x = x2.size()[-2] - x1.size()[-2]
            diff_y = x2.size()[-1] - x1.size()[-1]

            x1 = F.pad(x1, (diff_y, 0, diff_x, 0))
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(self.res_conv(x) + ft1)

        return r1


# ~~~~~~~~~~ RRU-Net ~~~~~~~~~~

class RRU_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RRU_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.GroupNorm(32, out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class RRU_first_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RRU_first_down, self).__init__()
        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_ch)
        )
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # the first ring conv
        # print(x.size())
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = torch.mul(1 + F.sigmoid(ft2), x)
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3


class RRU_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RRU_down, self).__init__()
        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch))
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False))

    def forward(self, x):
        x = self.pool(x)
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = torch.mul(1 + F.sigmoid(ft2), x)
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3


class RRU_up(nn.Module):
    def __init__(self, in_ch, out_ch, in_ch_skip=0, bilinear=False):
        super(RRU_up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            if in_ch_skip == 0:
                self.up = nn.Sequential(
                    nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2),
                    nn.GroupNorm(32, in_ch // 2)
                )
            else:
                self.up = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2),
                    nn.GroupNorm(32, in_ch)
                )

        self.conv = RRU_double_conv(in_ch + in_ch_skip, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch + in_ch_skip, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_ch))
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch + in_ch_skip, kernel_size=1, bias=False))

    def forward(self, x1, x2):
        # with torchsnooper.snoop():
        x1 = self.up(x1)
        diff_x = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diff_y, 0, diff_x, 0))  # 把x1变得和x2一样大

        x = self.relu(torch.cat([x2, x1], dim=1))

        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(self.res_conv(x) + ft1)
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = torch.mul(1 + F.sigmoid(ft2), x)
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3


# !!!!!!!!!!!! Universal functions !!!!!!!!!!!!

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class two_outconv(nn.Module):
    def __init__(self, in_ch):
        super(two_outconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.conv(x)
