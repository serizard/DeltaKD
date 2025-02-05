import os
import torch
import torch.distributed as dist
from timm.data import create_transform
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torchvision import datasets, transforms
from torchvision import datasets
import math

DATASET_STATS = {
    'cifar-100': {
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
        'num_classes': 100,
    },
    'cifar-10': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010),
        'num_classes': 10,
    },
    'imagenet-1k': {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'num_classes': 1000,
    },
    'imagenet-21k': {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'num_classes': 21843,
    },
}

class DatasetBuilder:
    def __init__(self, args):
        self.args = args
        self.distributed = args.distributed

    def build_transform(self, is_train=True):
        resize_im = self.args.input_size > 32
        if is_train:
            transform = create_transform(
                input_size=self.args.input_size,
                is_training=True,
                color_jitter=self.args.color_jitter,
                auto_augment=self.args.aa,
                interpolation=self.args.interpolation,
                re_prob=self.args.reprob,
                re_mode=self.args.remode,
                re_count=self.args.recount,
            )        
            if not resize_im:
                transform.transforms[0] = transforms.RandomCrop(
                    self.args.input_size, padding=4)
            return transform
        else:
            t = []
            if resize_im:
                size = int(self.args.input_size / self.args.eval_crop_ratio)
                t.append(
                    transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms.CenterCrop(self.args.input_size))

            t.append(transforms.ToTensor())
            if self.args.dataset in DATASET_STATS:
                t.append(transforms.Normalize(DATASET_STATS[self.args.dataset]['mean'], DATASET_STATS[self.args.dataset]['std']))
            else:
                raise ValueError(f"Dataset {self.args.dataset} is not supported")
            return transforms.Compose(t)

    def build_dataset(self, is_train=True):
        transform = self.build_transform(is_train)
        
        if self.args.dataset.startswith('cifar'):
            dataset_cls = datasets.CIFAR100 if self.args.dataset == 'cifar-100' else datasets.CIFAR10
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
        return dataset

    def build_loader(self, is_train=True):
        dataset = self.build_dataset(is_train)
        
        if self.distributed:
            if is_train:
                if self.args.repeated_aug:
                    sampler = RASampler(
                        dataset,
                        num_replicas=dist.get_world_size(),
                        rank=dist.get_rank(),
                        shuffle=True
                    )
                else:
                    sampler = DistributedSampler(
                        dataset, 
                        num_replicas=dist.get_world_size(),
                        rank=dist.get_rank(),
                        shuffle=True
                    )
            else:
                sampler = DistributedSampler(
                    dataset, 
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=False
                )
        else:
            sampler = torch.utils.data.RandomSampler(dataset)
            sampler = torch.utils.data.SequentialSampler(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            sampler=sampler,
            drop_last=is_train,
        )
        
        return dataloader

    @property
    def num_classes(self):
        return DATASET_STATS[self.args.dataset]['num_classes']


# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
class RASampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, num_repeats: int = 3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if num_repeats < 1:
            raise ValueError("num_repeats should be greater than 0")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * self.num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))

        # add extra samples to make it evenly divisible
        indices = torch.repeat_interleave(indices, repeats=self.num_repeats, dim=0).tolist()
        padding_size: int = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch