import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
import torch
import os

from torchvision.datasets.folder import IMG_EXTENSIONS


def get_transform(opt, method=transforms.InterpolationMode.BILINEAR, normalize=True):
    transform_list = []
    transform_list.append(transforms.Resize([opt.load_size, opt.load_size], method))


    if 'resize' in opt.resize_or_crop:
        transform_list.append(transforms.Resize([opt.load_size, opt.load_size], method))
    elif 'scale_short' in opt.resize_or_crop:
        transform_list.append(transforms.Resize(opt.load_size, method))

    if 'crop' in opt.resize_or_crop:
        transform_list.append(AlignedRandomCrop(opt.crop_size))

    if opt.is_train: # during testing, is_train is false. (may be in training it is also false)
        transform_list.append(AlignedRandomFlip(opt.flip_ratio))

    transform_list.append(transforms.ToTensor())
    if normalize:
        transform_list.append(transforms.Normalize(mean=0.5, std=0.5))

    return transforms.Compose(transform_list)

def is_image_file(filename): return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(_dir):
    images_paths = []
    assert os.path.isdir(_dir)

    for root, folder_name, filenames in sorted(os.walk(_dir)):
        for filename in filenames:
            if is_image_file(filename):
                images_paths.append(os.path.join(root, filename))

    return images_paths

class AlignedDataset(data.Dataset):

    def __init__(self, opt):
        super(AlignedDataset, self).__init__()

        self.opt = opt
        self.root = opt.data_root
        # layout path
        self.layout_paths = sorted(make_dataset(opt.layout_image_dir))
        # sem path
        self.sem_paths = sorted(make_dataset(opt.sem_image_dir))

        self.dataset_size = len(self.layout_paths)

    def __getitem__(self, index):
        # layout
        layout_path = self.layout_paths[index]
        layout = Image.open(layout_path).convert('L')
        transform = get_transform(self.opt)
        layout_tensor = transform(layout)

        # sem
        sem_path = self.sem_paths[index]
        sem_image = Image.open(sem_path).convert('L')
        sem_tensor = transform(sem_image)

        input_dict = {'layout': layout_tensor, 'sem': sem_tensor, 'path': layout_path}
        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batch_size * self.opt.batch_size

    @property
    def name(self):
        return 'AlignedDataset'

class AlignedRandomCrop(transforms.RandomCrop):
    def __init__(self, crop_size):
        super(AlignedRandomCrop, self).__init__(crop_size)
        self.count, self.seed = 0, torch.seed()

    def forward(self, image):
        if self.count ==  2: # for paired data
            self.count, self.seed = 0, torch.seed()  # get new random seed
        self.count += 1
        torch.manual_seed(self.seed)
        return super().forward(image)

class AlignedRandomFlip(transforms.RandomHorizontalFlip, transforms.RandomVerticalFlip):
    def __init__(self, flip_ratio):
        transforms.RandomHorizontalFlip.__init__(self, flip_ratio)
        transforms.RandomVerticalFlip.__init__(self, flip_ratio)
        self.count, self.seed = 0, torch.seed()

    def forward(self, image):
        if self.count == 2: # for paired data
            self.count, self.seed = 0, torch.seed()  # get new random seed
        self.count += 1
        torch.manual_seed(self.seed)
        image = transforms.RandomHorizontalFlip.forward(self, image)
        torch.manual_seed(self.seed)
        return transforms.RandomVerticalFlip.forward(self, image)





if __name__ == '__main__':
    from options.train_options import TrainOptions
    opt = TrainOptions().parse()
    dataset = AlignedDataset(opt)
    print(dataset.__getitem__(0))

