import os
import torchvision
import torch

from tqdm import tqdm

from data.data_loader import AlignedDatasetLoader
from options.test_options import TestOptions
from model.pixpix import Pix2Pix
from model.pixpix_hd_new_loss import Pix2PixHD
from model.mamba.mamba_gan import MambaGAN
from model.cfno import CFNOGAN
from model.damo import DAMOLitho
from model.doinn import DOINN
from model.lithogan import LithoGAN

from util.util import load_network, mkdir


if __name__=='__main__':
    opt = TestOptions().parse()
    opt.is_train = False # don't know why the fuck this paraser fuck me
    mkdir(opt.results_dir)

    data_loader = AlignedDatasetLoader(opt)
    dataset = data_loader.load_data()

    torch.cuda.set_device(opt.gpu_ids)
    # model = Pix2Pix(opt).to(opt.device)
    model = Pix2PixHD(opt).to(opt.device)
    # model = MambaGAN(opt).to(opt.device)
    load_network(model.net_g, './checkpoints/'+opt.name, 'G', 'latest')
    model.eval()

    image_id = 0
    for data in tqdm(dataset):
        if image_id >= opt.num_test:
            break
        layout, sem = data['layout'].to(opt.device), data['sem'].to(opt.device)

        predicted_sem = model.forward(layout)

        predicted_sem = predicted_sem.detach().cpu()

        if len(predicted_sem.shape) == 4:
            str = [ _.split('/')[-1].split('.')[-2] + '.jpg' for _ in data['path'] ]
            for i, s_i in zip(predicted_sem, str):
                torchvision.utils.save_image(i*0.5+0.5, os.path.join(opt.results_dir, s_i))
                image_id += 1
