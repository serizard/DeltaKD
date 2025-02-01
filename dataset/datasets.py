import os
import torch
import torch.distributed as dist
from timm.data import create_transform, Mixup
from timm.data.distributed_sampler import OrderedDistributedSampler
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets

from dataset.dataset_cfg import CIFAR10Config, CIFAR100Config, ImageNet1KConfig, ImageNet21KConfig  

class DatasetFactory:
    _DATASET_CONFIGS = {
        'cifar-100': CIFAR100Config,
        'cifar-10': CIFAR10Config,
        'imagenet-1k': ImageNet1KConfig,
        'imagenet-21k': ImageNet21KConfig,
    }

    @classmethod
    def get_config(cls, dataset_name):
        if dataset_name not in cls._DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}. "
                           f"Supported datasets are {list(cls._DATASET_CONFIGS.keys())}")
        return cls._DATASET_CONFIGS[dataset_name]()

class DatasetBuilder:
    def __init__(self, args):
        self.args = args
        self.config = DatasetFactory.get_config(args.dataset)
        self.distributed = args.distributed

    def build_transform(self, is_train=True):
        if is_train:
            transform = create_transform(
                input_size=self.config.INPUT_SIZE,
                is_training=True,
                auto_augment=f'rand-m{self.config.RAND_AUGMENT["num_ops"]}-'
                            f'mstd{self.config.RAND_AUGMENT["magnitude"]}',
                interpolation='bicubic',
                re_prob=self.config.RANDOM_ERASE['probability'],
                re_mode=self.config.RANDOM_ERASE['mode'],
                mean=self.config.MEAN,
                std=self.config.STD,
            )
        else:
            transform = create_transform(
                input_size=self.config.INPUT_SIZE,
                is_training=False,
                interpolation='bicubic',
                mean=self.config.MEAN,
                std=self.config.STD,
            )
        return transform

    def build_dataset(self, is_train=True):
        transform = self.build_transform(is_train)
        
        if self.config.NAME.startswith('cifar'):
            dataset_cls = datasets.CIFAR100 if self.config.NAME == 'cifar-100' else datasets.CIFAR10
            dataset = dataset_cls(
                root=self.args.data_path,
                train=is_train,
                transform=transform,
                download=True
            )
        else:
            split = 'train' if is_train else 'val'
            root = os.path.join(self.args.data_path, split)
            dataset = datasets.ImageFolder(root=root, transform=transform)
        
        # is_distributed = dist.is_initialized()
        # if is_train and is_distributed:
        #     rank = dist.get_rank()
        #     world_size = dist.get_world_size()
        #     total_size = len(dataset)
        #     indices = list(range(total_size))
        #     num_samples = total_size // world_size 
        #     start_index = num_samples * rank
        #     end_index = start_index + num_samples
        #     indices = indices[start_index:end_index] 
        #     dataset = torch.utils.data.Subset(dataset, indices) 

            
        return dataset

    def build_loader(self, is_train=True):
        dataset = self.build_dataset(is_train)
        
        if self.distributed:
            if is_train:
                sampler = DistributedSampler(
                    dataset, 
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=True
                )
            else:
                sampler = OrderedDistributedSampler(
                    dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank()
                )
            shuffle = False
        else:
            sampler = None
            shuffle = is_train

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            drop_last=is_train,
        )
        
        return dataloader, sampler

    def build_mixup_fn(self):
        mixup_fn = Mixup(
            mixup_alpha=self.config.MIXUP['mixup_alpha'],
            cutmix_alpha=self.config.MIXUP['cutmix_alpha'],
            cutmix_minmax=None,
            prob=self.config.MIXUP['prob'],
            switch_prob=self.config.MIXUP['switch_prob'],
            mode=self.config.MIXUP['mode'],
            label_smoothing=self.args.label_smoothing,
            num_classes=self.config.NUM_CLASSES
        )
        return mixup_fn

    @property
    def num_classes(self):
        return self.config.NUM_CLASSES