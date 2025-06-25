import torch
from vit_pytorch import ViT

from torch import nn
import torch.nn.functional as F
from model.network_module import GANLoss, NLayerDiscriminator

from model.trans.trans_G import get_b16_config
from model.trans.trans_G import VisionTransformer as Generator



class Trans(nn.Module):
    def __init__(self, opt):
        super(Trans, self).__init__()
        self.opt = opt

        config = get_b16_config()
        config.n_classes, config.n_skip = 1, 0
        self.net_g = Generator(config, img_size=opt.load_size, num_classes=1).to(opt.device)

        if opt.is_train:
            # self.net_d = ViT(
            #     image_size=opt.load_size,
            #     patch_size=32,
            #     num_classes=100,
            #     dim=1024,
            #     depth=6,
            #     heads=16,
            #     mlp_dim=2048,
            #     channels=2,
            #     dropout=0.1,
            #     emb_dropout=0.1
            # ).to(opt.device)

            self.net_d = NLayerDiscriminator(input_nc=2).to(opt.device)  # This is for region_loss/normal etc.
            self.criterionGAN = GANLoss(opt, use_lsgan=True).to(opt.device) # use_lsgan设置为True，使用MSELoss
            self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_d = torch.optim.Adam(self.net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.criterionL1 = nn.L1Loss().to(opt.device)

    def discriminate(self, layout, sem):
        return self.net_d(torch.cat((layout, sem), 1))

    def forward(self, x):
        return self.net_g(x)

    def backward_g(self, layout, fake_sem, sem):
        pred_fake = self.discriminate(layout, fake_sem)
        loss_g_fakeIsTrue = self.criterionGAN(pred_fake, True)
        loss_g_l1 = self.criterionL1(fake_sem, sem)
        return loss_g_fakeIsTrue + loss_g_l1


    def backward_d(self, layout, fake_sem, sem):
        pred_fake = self.discriminate(layout, fake_sem.detach())
        loss_d_fakeIsFake = self.criterionGAN(pred_fake, False)
        pred_true = self.discriminate(layout, sem)
        loss_d_trueIsTrue = self.criterionGAN(pred_true, True)
        return (loss_d_fakeIsFake + loss_d_trueIsTrue) / 2