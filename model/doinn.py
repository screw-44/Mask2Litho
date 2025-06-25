import torch
import torch.nn.functional as F
from einops import rearrange

from model.network_module import VanillaDiscriminator, GANLoss

from torch import nn

COMPLEXTYPE = torch.complex64

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

class RFNO(nn.Module):
    def __init__(self, out_channels, modes1, modes2):
        super(RFNO, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / out_channels)
        self.weights0 = nn.Parameter(self.scale * torch.rand(1, out_channels, 1, 1, dtype=COMPLEXTYPE))
        self.weights1 = nn.Parameter(self.scale * torch.rand(1, out_channels, self.modes1, self.modes2, dtype=COMPLEXTYPE))
        self.weights2 = nn.Parameter(self.scale * torch.rand(1, out_channels, self.modes1, self.modes2, dtype=COMPLEXTYPE))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        x_ft = x_ft * self.weights0

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=COMPLEXTYPE, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class RFNONet(nn.Module):
    def __init__(self):
        super().__init__()

        self.rfno = RFNO(64, modes1=32, modes2=32)

        self.conv0 = conv2d(1, 16, kernel_size=3, stride=2, padding=1, relu=True)
        self.conv1 = conv2d(16, 32, kernel_size=3, stride=2, padding=1, relu=True)
        self.conv2 = conv2d(32, 64, kernel_size=3, stride=2, padding=1, relu=True)

        self.deconv0 = deconv2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1, relu=True)
        self.deconv1 = deconv2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1, relu=True)
        self.deconv2 = deconv2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, relu=True)

        self.conv3 = conv2d(16, 16, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv4 = conv2d(16, 16, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv5 = conv2d(16, 8, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv6l = conv2d(8, 1, kernel_size=3, stride=1, padding=1, norm=False, relu=False)

    def forward(self, x):
        br0 = self.rfno(F.avg_pool2d(x, kernel_size=8, stride=8))

        br1_0 = self.conv0(x)
        br1_1 = self.conv1(br1_0)
        br1_2 = self.conv2(br1_1)

        joined = self.deconv0(torch.cat([br0, br1_2], dim=1))
        joined = self.deconv1(torch.cat([joined, br1_1], dim=1))
        joined = self.deconv2(torch.cat([joined, br1_0], dim=1))

        joined = self.conv3(joined)
        joined = self.conv4(joined)
        joined = self.conv5(joined)
        xl = self.conv6l(joined)

        return xl


class DOINN(nn.Module):
    def __init__(self, opt):
        super(DOINN, self).__init__()
        self.opt = opt
        self.net_g = RFNONet().to(opt.device)
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

