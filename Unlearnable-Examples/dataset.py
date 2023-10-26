import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import collections
import random

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Adjust transform options for MNIST
transform_options = {
    "MNIST": {
        "train_transform": [transforms.RandomRotation(10),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
}

@mlconfig.register
class DatasetGenerator():
    def __init__(self, train_batch_size=128, eval_batch_size=256, num_of_workers=4,
                 train_data_path='../datasets/', train_data_type='MNIST', seed=0,
                 test_data_path='../datasets/', test_data_type='MNIST'):
        np.random.seed(seed)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_of_workers = num_of_workers
        self.seed = seed
        self.train_data_type = train_data_type
        self.test_data_type = test_data_type
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

        train_transform = transform_options[train_data_type]['train_transform']
        test_transform = transform_options[test_data_type]['test_transform']
        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose(test_transform)

        # Training Datasets
        if train_data_type == 'MNIST':
            train_dataset = datasets.MNIST(root=train_data_path, train=True,
                                           download=True, transform=train_transform)
        else:
            raise('Training Dataset type %s not implemented' % train_data_type)

        # Test Dataset
        if test_data_type == 'MNIST':
            test_dataset = datasets.MNIST(root=test_data_path, train=False,
                                          download=True, transform=test_transform)
        else:
            raise('Test Dataset type %s not implemented' % test_data_type)

        self.datasets = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
        }
        return

    def getDataLoader(self, train_shuffle=True, train_drop_last=True):
        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=self.datasets['train_dataset'],
                                                   batch_size=self.train_batch_size,
                                                   shuffle=train_shuffle, pin_memory=True,
                                                   drop_last=train_drop_last, num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=self.datasets['test_dataset'],
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False, pin_memory=True,
                                                  drop_last=False, num_workers=self.num_of_workers)

        return data_loaders

# Add necessary modifications for the MNIST Poison Dataset if needed
class PoisonMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='samplewise', patch_location='center',
                 img_denoise=False, add_uniform_noise=False):
       
        pass

