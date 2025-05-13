import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.onnx.symbolic_opset9 import tensor

'''----------------------Generator----------------------'''
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        self.encoder1 = self.conv_stage(n_channels, 64)
        self.encoder2 = self.conv_stage(64, 128)
        self.encoder3 = self.conv_stage(128, 256)
        self.encoder4 = self.conv_stage(256, 512)

        self.bottleneck = self.conv_stage(512, 1024)

        self.up_conv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_stage(1024, 512)
        self.up_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_stage(512, 256)
        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_stage(256, 128)
        self.up_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_stage(128, 64)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    @staticmethod
    def conv_stage(in_channels, out_channels):
        stage = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return stage

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))

        bottle = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))

        dec4 = self.up_conv4(bottle)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.up_conv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.up_conv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.up_conv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.final_conv(dec1)

        return out

'''----------------------Discriminator----------------------'''
# FIXMe: Make this using unfold and fold
class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, patch_ratio=0.5, patch_times=2,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        """
        PatchGAN (Patch Generative Adversarial Network) is a discriminative model used commonly in the context of
        image-to-image translation tasks in Generative Adversarial Networks (GANs). Unlike typical GAN discriminators that
        output a single probability of whether the entire input image is real or fake, PatchGAN discriminators classify each
        N×N patch of the image independently, and average the results.
        :param input_nc:
        :param ndf: Number of filters in the first convolutional layer.  判别器中第一个卷积层的滤波器数量。
        :param n_layers:
        :param norm_layer:
        :param use_sigmoid:
        :param num_d:
        """
        super(PatchDiscriminator, self).__init__()
        self.patch_ratio, self.patch_times = patch_ratio, patch_times

        for i in range(self.patch_times):
            netd = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, False)
            setattr(self, 'layer' + str(i), netd.model)

    # input size should be [batch_size, 2(layout+sem), w, h]
    def forward(self, input):
        layer_inputs = [(input,), ]
        for _p in range(self.patch_times):
            last_patched_images = layer_inputs[-1]
            patch_images = []
            patch_image_width, patch_image_height = int(last_patched_images[0].size(2) * self.patch_ratio), int(last_patched_images[0].size(3) * self.patch_ratio)
            for image in last_patched_images: # iterate for each images of the patch layer
                for i in range(0, image.size(2), patch_image_width): # for each column
                    for j in range(0, image.size(3), patch_image_height): # for each row
                        patch_images.append(image[:, :, i:i+patch_image_width, j:j+patch_image_height])
            layer_inputs.append(patch_images)

        paired_results = []
        # iterate between discriminators
        for _p in range(self.patch_times): # get each layers of images
            model = getattr(self, 'layer' + str(_p))
            for patch in layer_inputs[_p]: # get each patch on each layer
                paired_results.append([patch, model(patch)])
        return paired_results

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_d=3, get_interm_feature=False):
        """
        PatchGAN (Patch Generative Adversarial Network) is a discriminative model used commonly in the context of
        image-to-image translation tasks in Generative Adversarial Networks (GANs). Unlike typical GAN discriminators that
        output a single probability of whether the entire input image is real or fake, PatchGAN discriminators classify each
        N×N patch of the image independently, and average the results.
        :param input_nc:
        :param ndf: Number of filters in the first convolutional layer.  判别器中第一个卷积层的滤波器数量。
        :param n_layers:
        :param norm_layer:
        :param use_sigmoid:
        :param num_d:
        :param get_interm_feature:
        """
        super(MultiscaleDiscriminator, self).__init__()
        self.num_d, self.n_layers, self.get_interm_feature = num_d, n_layers, get_interm_feature

        for i in range(num_d):
            netd = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, get_interm_feature)
            setattr(self, 'layer'+str(i), netd.model)
        self.down_sample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def single_d_forward(self, model, input):
        return [model(input)]

    def forward(self, input):
        result, input_downsampled = [], input
        # iterate between discriminators
        for i in range(self.num_d):
            model = getattr(self, 'layer'+str(self.num_d-i-1))
            result.append([model(input_downsampled),])
            if i != self.num_d - 1:
                input_downsampled = self.down_sample(input_downsampled)
        return result

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, get_interm_feature=False):
        super(NLayerDiscriminator, self).__init__()
        self.get_interm_feature, self.n_layers = get_interm_feature, n_layers

        kernel_size = 4
        pad_width = int((kernel_size - 1) // 2)
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=pad_width),]]

        nf = ndf
        for i in range(n_layers):
            prev_nf, nf = nf, min(nf*2, 512)
            stride = 2 if i != n_layers-1 else 1
            sequence.append([
                    nn.Conv2d(prev_nf, nf, kernel_size=kernel_size, stride=stride, padding=pad_width),
                    norm_layer(nf),
                    nn.LeakyReLU(0.2, True)
            ])
        sequence.append([nn.Conv2d(nf, 1, kernel_size=kernel_size, stride=1, padding=pad_width)])

        if use_sigmoid: sequence.append(nn.Sigmoid())

        if get_interm_feature:
            for n in range(len(sequence)):
                setattr(self, 'model_'+str(n), nn.Sequential(*sequence[n]))
        else:
            # HACK: this is for flatten the sequence, [[1, 2], [3, 4]] -> [1, 2, 3, 4]
            self.model = nn.Sequential(*sum(sequence, []))

    def forward(self, _input):
        if self.get_interm_feature:
            res = [_input]
            for n in range(self.n_layers+2):
                _model = getattr(self, 'model_'+str(n))
                res.append(_model(res[-1]))
            # return the results
            return res[1:]
        else:
            return self.model(_input)

class VanillaDiscriminator(nn.Module):
    def __init__(self, input_size, ndf=32, n_layers=2, norm_layer=nn.BatchNorm2d, use_sigmoid=False, get_interm_feature=False):
        super(VanillaDiscriminator, self).__init__()
        kernel_size = 4
        pad_width = int((kernel_size - 1) // 2)
        sequence = [[nn.Conv2d(2, ndf, kernel_size=kernel_size, stride=2, padding=pad_width), ]]

        nf = ndf
        for i in range(n_layers):
            prev_nf, nf = nf, min(nf * 2, 512)
            stride = 2
            sequence.append([
                nn.Conv2d(prev_nf, nf, kernel_size=kernel_size, stride=stride, padding=pad_width),
                nn.MaxPool2d(2, 2),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ])
        self.model = nn.Sequential(*sum(sequence, []))
        # O=（I-K+2P）/S+1， 输入图片（Input）大小为I*I，卷积核（Filter）大小为K*K，步长（stride）为S，填充（Padding）的像素数为P
        # 4 layers of conv layers, 1024 -> 512/256 -> 128/64 -> 32 ; batch_size: 2 -> 32 -> 64 -> 128
        # add linear layer
        self.flatten = nn.Flatten()
        self.Linear = nn.Linear(input_size//32*input_size//32*128, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, _input):
        conv_result = self.model(_input)
        out = self.flatten(conv_result)
        out = self.Linear(out)
        return out


'''----------------------Loss----------------------'''

class GANLoss(nn.Module):
    def __init__(self, opt, use_lsgan= True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label, self.fake_label = target_real_label, target_fake_label
        # can also be MSE loss， BCE的loss更加对称，MSE不是，同时生成出来的有奇怪的问题
        self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()
        self.opt = opt

    def get_target_tensor(self, _input, target_is_real):
        if target_is_real:
            return torch.tensor(self.real_label).expand_as(_input).to(self.opt.device)
        else:
            return torch.tensor(self.fake_label).expand_as(_input).to(self.opt.device)

    def __call__(self, input, target_is_real):
        """
        :param input: input is always with-in a list, and the last one is the latest result
        :param target_is_real:
        :return:
        """
        # When input is [[tensor, tensor], [tensor, tensor], may be is multi-batch scenario?
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1] # get the tensor out of the list
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
        # When input is [tensor, tensor, tensor]
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


if __name__ == '__main__':
    # Example usage:
    n_channels = 1  # Number of input channels (e.g., 3 for RGB images)
    n_classes = 3  # Number of output channels (e.g., 1 for binary segmentation)

    input = torch.randn(16, 1, 256, 256).cuda()
    model = NLayerDiscriminator(1).to('cuda')
    print(model)
    out = model(input)

    print(out.shape)