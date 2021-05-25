
from . import folder

import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def make_loaders(args, transforms, data_path, custom_class=None,
                 label_mapping=None, subset=None, subset_type='rand', subset_start=0,
                 only_val=False, seed=1, custom_class_args=None):
    '''
    **INTERNAL FUNCTION**
    This is an internal function that makes a loader for any dataset. You
    probably want to call dataset.make_loaders for a specific dataset,
    which only requires workers and batch_size. For example:
    >>> cifar_dataset = CIFAR10('/path/to/cifar')
    >>> train_loader, val_loader = cifar_dataset.make_loaders(workers=10, batch_size=128)
    >>> # train_loader and val_loader are just PyTorch dataloaders
    '''
    print(f"==> Preparing dataset {args.data_name}..")
    transform_train, transform_test = transforms
    if not args.data_aug:
        transform_train = transform_test

    val_batch_size = args.batch_size if not args.val_batch_size else args.val_batch_size

    if not custom_class:
        train_path = os.path.join(data_path, 'train')
        test_path = os.path.join(data_path, 'val')
        if not os.path.exists(test_path):
            test_path = os.path.join(data_path, 'test')

        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))

        if not only_val:
            train_set = folder.ImageFolder(root=train_path, transform=transform_train,
                                           label_mapping=label_mapping)
        test_set = folder.ImageFolder(root=test_path, transform=transform_test,
                                      label_mapping=label_mapping)
    else:
        if custom_class_args is None: custom_class_args = {}
        if not only_val:
            train_set = custom_class(root=data_path, train=True, download=True,
                                     transform=transform_train, **custom_class_args)
        test_set = custom_class(root=data_path, train=False, download=True,
                                transform=transform_test, **custom_class_args)

    if not only_val:
        attrs = ["samples", "train_data", "data"]
        vals = {attr: hasattr(train_set, attr) for attr in attrs}
        assert any(vals.values()), f"dataset must expose one of {attrs}"
        train_sample_count = len(getattr(train_set, [k for k in vals if vals[k]][0]))

    if (not only_val) and (subset is not None) and (subset <= train_sample_count):
        assert not only_val
        if subset_type == 'rand':
            rng = np.random.RandomState(seed)
            subset = rng.choice(list(range(train_sample_count)), size=subset+subset_start, replace=False)
            subset = subset[subset_start:]
        elif subset_type == 'first':
            subset = np.arange(subset_start, subset_start + subset)
        else:
            subset = np.arange(train_sample_count - subset, train_sample_count)

        train_set = Subset(train_set, subset)
    if args.distributed :
        world_size = dist.get_world_size()
    train_sampler = DistributedSampler(train_set) if args.distributed else None
    test_sampler = DistributedSampler(test_set) if args.distributed else None
    if not only_val:
        train_bs = int(args.batch_size / world_size) if args.distributed else args.batch_size
        train_loader = DataLoader(train_set,
                                  batch_size=train_bs,
                                  shuffle=(train_sampler is None),
                                  num_workers=args.workers,
                                  sampler=train_sampler,
                                  pin_memory=True)
    val_bs = train_bs = int(val_batch_size / world_size) if args.distributed else val_batch_size
    test_loader = DataLoader(test_set,
                             sampler=test_sampler,
                             batch_size=val_bs,
                             shuffle=False,
                             num_workers=args.workers,
                             pin_memory=True)

    if only_val:
        return None, test_loader

    return train_loader, test_loader, train_sampler
