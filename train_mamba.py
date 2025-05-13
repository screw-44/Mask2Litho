import numpy as np
import os
import time
import torchvision
import torch
import torch.nn.functional as F

from tqdm import trange, tqdm

from util.util import lcm, save_network, load_network
from options.train_options import TrainOptions
from data.data_loader import AlignedDatasetLoader
from model.pixpix_hd_new_loss import Pix2PixHD
from model.mamba.mamba_gan import MambaGAN

opt = TrainOptions().parse()

# calculate lcm to ensure the evenly distributed log during training
opt.print_freq = lcm(opt.print_freq, opt.batch_size)

torch.cuda.set_device(opt.gpu_ids)
data_loader = AlignedDatasetLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print("# training dataset size:{}".format(dataset_size))

model = MambaGAN(opt).to(opt.device)

# custom end
experiment_path = str(os.path.join(opt.checkpoints_dir, opt.name))
iter_path = str(os.path.join(experiment_path, 'iter.txt'))
if opt.continue_train:
    start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    print("Resuming from epoch {} at iteration {}".format(start_epoch, epoch_iter))

if not opt.is_train or opt.continue_train:
    load_network(model.net_g, experiment_path, 'G', opt.which_epoch)
    if opt.is_train:
        load_network(model.net_d, experiment_path, 'D', opt.which_epoch)
else:
    start_epoch, epoch_iter = 1, 0

for epoch in range(start_epoch, 500 + 1):
    scaler = torch.amp.GradScaler('cuda')
    for data in tqdm(dataset):
        # set up input
        with torch.autocast(device_type=opt.device, dtype=torch.float16):
            layout, sem = data['layout'].to(opt.device), data['sem'].to(opt.device)
            # inference
            fake_sem = model.forward(layout)
        # backpropagation for D, enable backprop for D
        for param in model.net_d.parameters():
            param.requires_grad = True

        # loss in edge region
        edge_threshold = 240/255
        edge_fake_sem, edge_real_sem = fake_sem.clone(), sem.clone()
        edge_fake_sem[edge_fake_sem<edge_threshold], edge_real_sem[edge_real_sem<edge_threshold] = 0, 0
        edge_loss = model.criterionL1(edge_fake_sem, edge_real_sem)
        
        d_loss = model.backward_d(layout, fake_sem, sem)
        # d_loss += edge_loss
        scaler.scale(d_loss).backward()
        scaler.step(model.optimizer_d)
        model.optimizer_d.zero_grad()
        scaler.update()


        # D requires no gradients when optimizing G
        for param in model.net_d.parameters():
            param.requires_grad = False
        g_loss = model.backward_g(layout, fake_sem, sem)
        g_loss += edge_loss
        scaler.scale(g_loss).backward()
        scaler.step(model.optimizer_g)
        model.optimizer_g.zero_grad()
        scaler.update()


    print('Epoch: {} has g_loss: {}, d_loss: {}'.format(epoch, g_loss, d_loss))
    if epoch % opt.display_freq == 0:
        print('saving the model and image at epoch: {}'.format(epoch))
        save_network(model.net_g, experiment_path, 'G', epoch)
        save_network(model.net_d, experiment_path, 'D', epoch)

        # save latest
        save_network(model.net_g, experiment_path, 'G', 'latest')
        save_network(model.net_d, experiment_path, 'D', 'latest')

        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        show_image = torch.cat((layout, fake_sem*0.5+0.5, sem*0.5+0.5), dim=0)
        grid = torchvision.utils.make_grid(show_image, opt.batch_size)
        torchvision.utils.save_image(grid, os.path.join(experiment_path, 'epoch_' + str(epoch) + '.png'))


