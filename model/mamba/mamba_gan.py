

import torch
import torch.nn as nn

from model.mamba.mamba_d import MambaD
from model.mamba.networks.vision_mamba import MambaUnet

from model.network_module import NLayerDiscriminator, GANLoss, VanillaDiscriminator

class MambaGAN(nn.Module):
    def __init__(self, opt):
        super(MambaGAN, self).__init__()
        self.opt = opt


        self.net_g = MambaUnet(opt.load_size, 1).to(opt.device)
        self.net_g.load_from('./pretrained_ckpt/vmamba_tiny_e292.pth') # comment说必须要有pre-trained的作为载入值，和transformer类似

        if opt.is_train:
            # self.net_d = MambaD(opt, input_size=opt.load_size, in_chans=2).to(opt.device) # This is for MambaDiscriminator
            self.net_d = NLayerDiscriminator(input_nc=2).to(opt.device) # This is for region_loss/normal etc.
            # self.net_d = VanillaDiscriminator(opt.load_size).to(opt.device)             # valila discriminator for baseline compare
            # 判别器先用之前？ 对，然后抄 一下Pix2PixHD_newloss实现一下新的 网络结构（未来考虑下把discriminator也换成Vmamba）
            self.criterionGAN = GANLoss(opt, use_lsgan=True).to(opt.device) # use_lsgan设置为False，使用BCELoss，MSELoss生成太平滑了
            self.criterionL1 = nn.L1Loss().to(opt.device)

            self.CriterionCos = nn.CosineSimilarity().to(opt.device)
            self.CriterionKL = nn.KLDivLoss().to(opt.device)

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

        loss_g_l1 = self.criterionL1(fake_sem, sem)
        return loss_g_fakeIsTrue + loss_g_l1 * 1

    def backward_d(self, layout, fake_sem, sem):
        pred_fake = self.discriminate(layout, fake_sem.detach())
        loss_d_fakeIsFake = self.criterionGAN(pred_fake, False)

        pred_true = self.discriminate(layout, sem)
        loss_d_trueIsTrue = self.criterionGAN(pred_true, True)

        return (loss_d_fakeIsFake + loss_d_trueIsTrue) * 0.1

if __name__ == '__main__':
    from options.train_options import TrainOptions
    opt = TrainOptions().parse()
    mamba = MambaGAN(opt)

    layout = torch.ones(8, 1, 256, 256).to(opt.device)
    sem = torch.ones(8, 1, 256, 256).to(opt.device) * 5 + 4

    for i in range(100):
        fake_sem = mamba(layout)
        loss_g = mamba.backward_g(layout, fake_sem, sem)
        loss_g.backward()
        mamba.optimizer_g.step()
        mamba.optimizer_g.zero_grad()

        loss_d = mamba.backward_d(layout, fake_sem, sem)
        loss_d.backward()
        mamba.optimizer_g.step()
        mamba.optimizer_g.zero_grad()

        print("Loss is g:%s, d:%s" % (loss_g, loss_d))


