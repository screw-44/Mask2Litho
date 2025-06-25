from torch.utils.data import DataLoader
from data.dataset import AlignedDataset


class AlignedDatasetLoader:
    def __init__(self, opt):
        self.opt = opt
        self.dataset = AlignedDataset(opt)
        print("dataset {} was created".format(self.dataset.name))
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.opt.batch_size,
            shuffle=self.opt.shuffle,
            num_workers=self.opt.num_workers,
        )

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    @staticmethod
    def name():
        return "AlignedDatasetLoader"

    def load_data(self):
        return self.dataloader
