import torch
import torch.nn as nn

from model.mamba.networks.mamba_sys import PatchEmbed2D, VSSLayer, PatchMerging2D, PatchExpand, VSSLayer_up
from util.util import model_summary

class MambaD(nn.Module):
    def __init__(self, opt, input_size, patch_size=4, in_chans=1, depths=[2, 2, 4, 2], drop_path_rate=0.1,
                 dims=[16, 32, 64, 128], d_state=16, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False):
        super(MambaD, self).__init__()
        self.opt = opt
        self.num_layers = len(depths)
        self.embed_dim = dims[0]
        self.num_features = dims[-1]

        self.norm = norm_layer(self.num_features)

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim, norm_layer=norm_layer if patch_norm else None)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # actually network structure here(encoder)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                # dim = dims[i_layer],
                dim=int(dims[0] * 2 ** i_layer),
                depth = depths[i_layer],
                d_state=dims[0] // 6 if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[:i_layer+1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)


        self.conv = nn.Conv2d(128, 1, (3,3), stride=1, padding=1)


    def forward(self, x):
        # 先4倍降采样 conv
        x = self.patch_embed(x)
        down_samples = []
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # 转回 batch, c, h, w的形式
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        return x

if __name__ == "__main__":

    input = torch.randn(2, 2, 1024, 1024).cuda()
    model = MambaD(opt=None, input_size=1024, in_chans=2).to('cuda')
    out = model(input)
    print(out.shape)
    print(model_summary(model))