import argparse
import torch
import os

from util.util import mkdirs


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='DefaultExperimentName',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument("--is_train", type=bool, default=True)
        self.parser.add_argument("--device", type=str, default='mps:0')
        self.parser.add_argument('--gpu_ids', type=int, default=3,
                                 help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                                 help='models are saved here')
        self.parser.add_argument('--model', type=str, default='pix2pixHD',
                                 help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true',
                                 help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32],
                                 help="Supported data type i.e. 8, 16, 32 bit")
        """ Verbose mode: 详细模式. Terminal output more detailed information. """
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--fp16', action='store_true', default=True, help='train with AMP')
        """ local_rank: the rank of the process on the local machine. Used for multi-node training """
        self.parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

        # for data
        self.parser.add_argument('--layout_image_dir', type=str, default='../sjw/layout2adi0907/train/layout', help='the directory of label files')
        self.parser.add_argument('--sem_image_dir', type=str, default='../sjw/layout2adi0907/train/ADI', help='the directory of real images')
        self.parser.add_argument('--shuffle', action='store_true', help='whether to shuffle training data')
        self.parser.add_argument('--num_workers', type=int, default=16, help='number of worker processes')

        # input/output sizes
        self.parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        self.parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--flip_ratio', type=float, default=0.5, help='image flip percentage')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--data_root', type=str, default='../sjw/layout2adi0927/')
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_short_and_crop',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_short|scale_short_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512, help='display window size')
        self.parser.add_argument('--tf_log', action='store_true',
                                 help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4,
                                 help='number of down-sampling layers in netG')
        self.parser.add_argument('--n_blocks_global', type=int, default=9,
                                 help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3,
                                 help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        self.parser.add_argument('--niter_fix_global', type=int, default=0,
                                 help='number of epochs that we only train the out-most local enhancer')

        # for instance-wise features
        self.parser.add_argument('--no_instance', action='store_true',
                                 help='if specified, do *not* add instance map as input')
        self.parser.add_argument('--instance_feat', action='store_true',
                                 help='if specified, add encoded instance features as input')
        self.parser.add_argument('--label_feat', action='store_true',
                                 help='if specified, add encoded label features as input')
        self.parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')
        self.parser.add_argument('--load_features', action='store_true',
                                 help='if specified, load precomputed feature maps')
        self.parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder')
        self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        self.parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        # str_ids = self.opt.gpu_ids.split(',')
        # self.opt.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         self.opt.gpu_ids.append(id)

        # set gpu ids
        # if len(self.opt.gpu_ids) > 0:
        #     torch.cuda.set_device(self.opt.gpu_ids[0])

        if torch.cuda.is_available():
            self.opt.device = 'cuda'
        else:
            self.opt.device = 'mps:0'

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        mkdirs(expr_dir)
        if save:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt


if __name__ == '__main__':
    option = BaseOptions()
