
import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange

from model.network_module import VanillaDiscriminator, GANLoss


def conv2d(chIn, chOut, kernel_size, stride, padding, bias=True, norm=True, relu=False):
    layers = []
    layers.append(nn.Conv2d(chIn, chOut, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    if norm:
        layers.append(nn.BatchNorm2d(chOut, affine=bias))
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def deconv2d(chIn, chOut, kernel_size, stride, padding, output_padding, bias=True, norm=True, relu=False):
    layers = []
    layers.append(nn.ConvTranspose2d(chIn, chOut, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias))
    if norm:
        layers.append(nn.BatchNorm2d(chOut, affine=bias))
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def sepconv2d(chIn, chOut, kernel_size, stride, padding, bias=True, norm=True, relu=False):
    layers = []
    layers.append(nn.Conv2d(chIn, chOut, groups=chIn, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    if norm:
        layers.append(nn.BatchNorm2d(chOut, affine=bias))
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def repeat2d(n, chIn, chOut, kernel_size, stride, padding, bias=True, norm=True, relu=False):
    layers = []
    for idx in range(n):
        layers.append(nn.Conv2d(chIn if idx == 0 else chOut, chOut, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(chOut, affine=bias))
        if relu:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def spsr(r, chIn, chOut, kernel_size, stride, padding, bias=True, norm=True, relu=False):
    layers = []
    layers.append(nn.Conv2d(chIn, chOut*(r**2), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    layers.append(nn.PixelShuffle(r))
    if norm:
        layers.append(nn.BatchNorm2d(chOut, affine=bias))
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def linear(chIn, chOut, bias=True, norm=True, relu=False):
    layers = []
    layers.append(nn.Linear(chIn, chOut, bias=bias))
    if norm:
        layers.append(nn.BatchNorm1d(chOut, affine=bias))
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def split(x, size=16):
    return rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=size, s2=size)

class CFNO(nn.Module):
    def __init__(self, c=1, d=16, k=16, s=1, size=(128, 128)):
        super().__init__()
        self.c = c
        self.d = d
        self.k = k
        self.s = s
        self.size = size
        self.fc = nn.Linear(self.c*(self.k**2), self.d, dtype=torch.complex64)
        self.conv = sepconv2d(self.d, self.d, kernel_size=2*self.s+1, stride=1, padding="same", relu=False)

    def forward(self, x):
        batchsize = x.shape[0]
        c = x.shape[1]
        h = x.shape[2]//self.k
        w = x.shape[3]//self.k
        patches = split(x, self.k)
        patches = patches.view(-1, self.c*(self.k**2))
        fft = torch.fft.fft(patches, dim=-1)
        fc = self.fc(fft)
        ifft = torch.fft.ifft(fc).real
        ifft = rearrange(ifft, '(b h w) d -> b d h w', h=h, w=w)
        conved = self.conv(ifft)
        return F.interpolate(conved, size=self.size)

class CFNONet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cfno0 = CFNO(c=1, d=16, k=16, s=1)
        self.cfno1 = CFNO(c=1, d=32, k=32, s=1)
        self.cfno2 = CFNO(c=1, d=64, k=64, s=1)

        self.conv0a = conv2d(1, 32, kernel_size=3, stride=2, padding=1, relu=True)
        self.conv0b = repeat2d(2, 32, 32, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv1a = conv2d(32, 64, kernel_size=3, stride=2, padding=1, relu=True)
        self.conv1b = repeat2d(2, 64, 64, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv2a = conv2d(64, 128, kernel_size=3, stride=2, padding=1, relu=True)
        self.conv2b = repeat2d(2, 128, 128, kernel_size=3, stride=1, padding=1, relu=True)
        self.branch = nn.Sequential(self.conv0a, self.conv0b, self.conv1a, self.conv1b, self.conv2a, self.conv2b)

        self.deconv0a = deconv2d(16 + 32 + 64 + 128, 128, kernel_size=3, stride=2, padding=1, output_padding=1,
                                 relu=True)
        self.deconv0b = repeat2d(2, 128, 128, kernel_size=3, stride=1, padding=1, relu=True)
        self.deconv1a = deconv2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, relu=True)
        self.deconv1b = repeat2d(2, 64, 64, kernel_size=3, stride=1, padding=1, relu=True)
        self.deconv2a = deconv2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, relu=True)
        self.deconv2b = repeat2d(2, 32, 32, kernel_size=3, stride=1, padding=1, relu=True)

        self.conv3 = conv2d(32, 32, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv4 = conv2d(32, 32, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv5 = conv2d(32, 32, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv6l = conv2d(32, 1, kernel_size=3, stride=1, padding=1, norm=False, relu=False)

        self.tail = nn.Sequential(self.deconv0a, self.deconv0b, self.deconv1a, self.deconv1b, self.deconv2a,
                                  self.deconv2b,
                                  self.conv3, self.conv4, self.conv5)

    def forward(self, x):
        br0 = self.cfno0(x)
        br1 = self.cfno1(x)
        br2 = self.cfno2(x)
        br3 = self.branch(x)

        feat = torch.cat([br0, br1, br2, br3], dim=1)
        x = self.tail(feat)
        xl = self.conv6l(x)

        return xl

class CFNOGAN(nn.Module):
    def __init__(self, opt):
        super(CFNOGAN, self).__init__()
        self.opt = opt
        self.net_g = CFNONet().to(opt.device)
        if opt.is_train:
            self.net_d = VanillaDiscriminator(input_size=1024).to(opt.device)
            self.criterionGAN = GANLoss(opt, use_lsgan=True).to(opt.device) # use_lsgan设置为True，使用MSELoss
            self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_d = torch.optim.Adam(self.net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def discriminate(self, layout, sem):
        # note: for discriminator, sem must be sem.detach()
        return self.net_d(torch.cat((layout, sem), 1))

    def forward(self, layout):
        return self.net_g(layout)

    def backward_g(self, layout, fake_sem, sem):
        pred_fake = self.discriminate(layout, fake_sem)
        loss_g_fakeIsTrue = self.criterionGAN(pred_fake, True)
        return loss_g_fakeIsTrue

    def backward_d(self, layout, fake_sem, sem):
        pred_fake = self.discriminate(layout, fake_sem.detach())
        loss_d_fakeIsFake = self.criterionGAN(pred_fake, False)
        pred_true = self.discriminate(layout, sem)
        loss_d_trueIsTrue = self.criterionGAN(pred_true, True)
        return (loss_d_fakeIsFake + loss_d_trueIsTrue) / 2