import torch as ch
import torch.utils.data
from torchvision import datasets
from . import constants
from . import data_augmentation as da
from . import loaders
from src.datasets.helpers import get_label_mapping
import os
import pickle
import logging

import numpy as np

from torch.utils.data import Sampler

from . import folder


class DataSet(object):
    '''
    Base class for representing a dataset. Meant to be subclassed, with
    subclasses implementing the `get_model` function.
    '''

    def __init__(self, ds_name, data_path, **kwargs):
        """
        Args:
            ds_name (str) : string identifier for the dataset
            data_path (str) : path to the dataset
            num_classes (int) : *required kwarg*, the number of classes in
                the dataset
            mean (ch.tensor) : *required kwarg*, the mean to normalize the
                dataset with (e.g.  :samp:`ch.tensor([0.4914, 0.4822,
                0.4465])` for CIFAR-10)
            std (ch.tensor) : *required kwarg*, the standard deviation to
                normalize the dataset with (e.g. :samp:`ch.tensor([0.2023,
                0.1994, 0.2010])` for CIFAR-10)
            custom_class (type) : *required kwarg*, a
                :samp:`torchvision.models` class corresponding to the
                dataset, if it exists (otherwise :samp:`None`)
            label_mapping (dict[int,str]) : *required kwarg*, a dictionary
                mapping from class numbers to human-interpretable class
                names (can be :samp:`None`)
            transform_train (torchvision.transforms) : *required kwarg*,
                transforms to apply to the training images from the
                dataset
            transform_test (torchvision.transforms) : *required kwarg*,
                transforms to apply to the validation images from the
                dataset
        """
        required_args = ['num_classes', 'mean', 'std',
                         'transform_train', 'transform_test']
        optional_args = ['custom_class', 'label_mapping', 'custom_class_args']

        missing_args = set(required_args) - set(kwargs.keys())
        if len(missing_args) > 0:
            raise ValueError("Missing required args %s" % missing_args)

        extra_args = set(kwargs.keys()) - set(required_args + optional_args)
        if len(extra_args) > 0:
            raise ValueError("Got unrecognized args %s" % extra_args)
        final_kwargs = {k: kwargs.get(k, None) for k in required_args + optional_args}

        self.ds_name = ds_name
        self.data_path = data_path
        self.__dict__.update(final_kwargs)

    def override_args(self, default_args, kwargs):
        '''
        Convenience method for overriding arguments. (Internal)
        '''
        for k in kwargs:
            if not (k in default_args): continue
            req_type = type(default_args[k])
            no_nones = (default_args[k] is not None) and (kwargs[k] is not None)
            if no_nones and (not isinstance(kwargs[k], req_type)):
                raise ValueError(f"Argument {k} should have type {req_type}")
        return {**default_args, **kwargs}


    def make_loaders(self, args, subset=None, subset_start=0, subset_type='rand',
                     only_val=False, subset_seed=None):
        '''
        Args:
            workers (int) : number of workers for data fetching (*required*).
                batch_size (int) : batch size for the data loaders (*required*).
            data_aug (bool) : whether or not to do train data augmentation.
            subset (None|int) : if given, the returned training data loader
                will only use a subset of the training data; this should be a
                number specifying the number of training data points to use.
            subset_start (int) : only used if `subset` is not None; this specifies the
                starting index of the subset.
            subset_type ("rand"|"first"|"last") : only used if `subset is
                not `None`; "rand" selects the subset randomly, "first"
                uses the first `subset` images of the training data, and
                "last" uses the last `subset` images of the training data.
            seed (int) : only used if `subset == "rand"`; allows one to fix
                the random seed used to generate the subset (defaults to 1).
            val_batch_size (None|int) : if not `None`, specifies a
                different batch size for the validation set loader.
            only_val (bool) : If `True`, returns `None` in place of the
                training data loader
            shuffle_train (bool) : Whether or not to shuffle the training data
                in the returned DataLoader.
            shuffle_val (bool) : Whether or not to shuffle the test data in the
                returned DataLoader.
        Returns:
            A training loader and validation loader according to the
            parameters given. These are standard PyTorch data loaders, and
            thus can just be used via:
            >>> train_loader, val_loader = ds.make_loaders(workers=8, batch_size=128)
            >>> for im, lab in train_loader:
            >>>     # Do stuff...
        '''
        transforms = (self.transform_train, self.transform_test)
        return loaders.make_loaders(args=args,
                                    transforms=transforms,
                                    data_path=self.data_path,
                                    label_mapping=self.label_mapping,
                                    custom_class=self.custom_class,
                                    subset=subset,
                                    subset_start=subset_start,
                                    subset_type=subset_type,
                                    only_val=only_val,
                                    seed=subset_seed,
                                    custom_class_args=self.custom_class_args)


class ImageNet(DataSet):
    '''
    ImageNet Dataset [DDS+09]_.
    Requires ImageNet in ImageFolder-readable format.
    ImageNet can be downloaded from http://www.image-net.org. See
    `here <https://pytorch.org/docs/master/torchvision/datasets.html#torchvision.datasets.ImageFolder>`_
    for more information about the format.
    .. [DDS+09] Deng, J., Dong, W., Socher, R., Li, L., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. 2009 IEEE Conference on Computer Vision and Pattern Recognition, 248-255.
    '''
    def __init__(self, data_path, **kwargs):
        """
        """
        ds_kwargs = {
            'num_classes': 1000,
            'mean': ch.tensor([0.485, 0.456, 0.406]),
            'std': ch.tensor([0.229, 0.224, 0.225]),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': da.TRAIN_TRANSFORMS_IMAGENET,
            'transform_test': da.TEST_TRANSFORMS_IMAGENET
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(ImageNet, self).__init__('imagenet', data_path, **ds_kwargs)



class RestrictedImageNet(DataSet):
    '''
    RestrictedImagenet Dataset [TSE+19]_
    A subset of ImageNet with the following labels:
    * Dog (classes 151-268)
    * Cat (classes 281-285)
    * Frog (classes 30-32)
    * Turtle (classes 33-37)
    * Bird (classes 80-100)
    * Monkey (classes 365-382)
    * Fish (classes 389-397)
    * Crab (classes 118-121)
    * Insect (classes 300-319)
    To initialize, just provide the path to the full ImageNet dataset
    (no special formatting required).
    .. [TSE+19] Tsipras, D., Santurkar, S., Engstrom, L., Turner, A., &
        Madry, A. (2019). Robustness May Be at Odds with Accuracy. ICLR
        2019.
    '''
    def __init__(self, data_path, **kwargs):
        """
        """
        ds_name = 'restricted_imagenet'
        ds_kwargs = {
            'num_classes': len(constants.RESTRICTED_IMAGNET_RANGES),
            'mean': ch.tensor([0.4717, 0.4499, 0.3837]),
            'std': ch.tensor([0.2600, 0.2516, 0.2575]),
            'custom_class': None,
            'label_mapping': get_label_mapping(ds_name, constants.RESTRICTED_IMAGNET_RANGES),
            'transform_train': da.TRAIN_TRANSFORMS_IMAGENET,
            'transform_test': da.TEST_TRANSFORMS_IMAGENET
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(RestrictedImageNet, self).__init__(ds_name, data_path, **ds_kwargs)



class CustomImageNet(DataSet):
    '''
    CustomImagenet Dataset
    A subset of ImageNet with the user-specified labels
    To initialize, just provide the path to the full ImageNet dataset
    along with a list of lists of wnids to be grouped together
    (no special formatting required).
    '''
    def __init__(self, data_path, custom_grouping, **kwargs):
        """
        """
        ds_name = 'custom_imagenet'
        ds_kwargs = {
            'num_classes': len(custom_grouping),
            'mean': ch.tensor([0.4717, 0.4499, 0.3837]),
            'std': ch.tensor([0.2600, 0.2516, 0.2575]),
            'custom_class': None,
            'label_mapping': get_label_mapping(ds_name, custom_grouping),
            'transform_train': da.TRAIN_TRANSFORMS_IMAGENET,
            'transform_test': da.TEST_TRANSFORMS_IMAGENET
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(CustomImageNet, self).__init__(ds_name, data_path, **ds_kwargs)



class CIFAR10(DataSet):
    """
    CIFAR-10 dataset [Kri09]_.
    A dataset with 50k training images and 10k testing images, with the
    following classes:
    * Airplane
    * Automobile
    * Bird
    * Cat
    * Deer
    * Dog
    * Frog
    * Horse
    * Ship
    * Truck
    .. [Kri09] Krizhevsky, A (2009). Learning Multiple Layers of Features
        from Tiny Images. Technical Report.
    """
    def __init__(self, data_path='/tmp/', **kwargs):
        """
        """
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.4914, 0.4822, 0.4465]),
            'std': ch.tensor([0.2023, 0.1994, 0.2010]),
            'custom_class': datasets.CIFAR10,
            'label_mapping': None,
            'transform_train': da.TRAIN_TRANSFORMS_DEFAULT(32),
            'transform_test': da.TEST_TRANSFORMS_DEFAULT(32)
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(CIFAR10, self).__init__('cifar', data_path, **ds_kwargs)



class CIFAR100(DataSet):
    """
    CIFAR-10 dataset [Kri09]_.
    A dataset with 50k training images and 10k testing images, with the
    following classes:
    * Airplane
    * Automobile
    * Bird
    * Cat
    * Deer
    * Dog
    * Frog
    * Horse
    * Ship
    * Truck
    .. [Kri09] Krizhevsky, A (2009). Learning Multiple Layers of Features
        from Tiny Images. Technical Report.
    """
    def __init__(self, data_path='/tmp/', **kwargs):
        """
        """
        ds_kwargs = {
            'num_classes': 100,
            'mean': ch.tensor([0.4914, 0.4822, 0.4465]),
            'std': ch.tensor([0.2023, 0.1994, 0.2010]),
            'custom_class': datasets.CIFAR100,
            'label_mapping': None,
            'transform_train': da.TRAIN_TRANSFORMS_DEFAULT(32),
            'transform_test': da.TEST_TRANSFORMS_DEFAULT(32)
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(CIFAR100, self).__init__('cifar', data_path, **ds_kwargs)


class MNIST(DataSet):
    """
    CIFAR-10 dataset [Kri09]_.
    A dataset with 50k training images and 10k testing images, with the
    following classes:
    * Airplane
    * Automobile
    * Bird
    * Cat
    * Deer
    * Dog
    * Frog
    * Horse
    * Ship
    * Truck
    .. [Kri09] Krizhevsky, A (2009). Learning Multiple Layers of Features
        from Tiny Images. Technical Report.
    """
    def __init__(self, data_path='/tmp/', **kwargs):
        """
        """
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.4914, 0.4822, 0.4465]),
            'std': ch.tensor([0.2023, 0.1994, 0.2010]),
            'custom_class': datasets.MNIST,
            'label_mapping': None,
            'transform_train': da.TRAIN_TRANSFORMS_MNIST,
            'transform_test': da.TEST_TRANSFORMS_MNIST
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(MNIST, self).__init__('mnist', data_path, **ds_kwargs)



DATASETS = {
    'imagenet': ImageNet,
    'restricted_imagenet': RestrictedImageNet,
    'custom_imagenet': CustomImageNet,
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'mnist': MNIST,
}
'''
Dictionary of datasets. A dataset class can be accessed as:
>>> import robustness.datasets
>>> ds = datasets.DATASETS['cifar']('/path/to/cifar')
'''


def get_dataset(args):
    return DATASETS[args.data_name](data_path=args.data_dir)

class SemiSupervisedDataset(DataSet):
    def __init__(self,
                 args,
                 take_amount=None,
                 take_amount_seed=13,
                 add_svhn_extra=False,
                 aux_data_filename=None,
                 add_aux_labels=False,
                 aux_take_amount=None,
                 train=False,
                 **kwargs):
        """A dataset with auxiliary pseudo-labeled data"""

        dataset = get_dataset(args)

        if not dataset.custom_class:
            if train :
                data_path = os.path.join(args.data_dir, 'train')
                self.dataset = folder.ImageFolder(root=data_path, transform=dataset.transform_train,
                                               label_mapping=args.label_mapping)
            else :
                data_path = os.path.join(args.data_dir, 'val')
                self.dataset = folder.ImageFolder(root=data_path, transform=dataset.transform_test,
                                              label_mapping=dataset.label_mapping)
            if not os.path.exists(data_path):
                data_path = os.path.join(args.data_dir, 'test')

            if not os.path.exists(data_path):
                raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))
        else:
            if dataset.custom_class_args is None: dataset.custom_class_args = {}
            if train:
                custom_class_args = dataset.custom_class_args
                data_path = os.path.join(args.data_dir)
                self.dataset = dataset.custom_class(root=data_path, train=True, download=True,
                                         transform=dataset.transform_train, **custom_class_args)
            else :
                custom_class_args = dataset.custom_class_args
                data_path = os.path.join(args.data_dir)
                self.dataset = dataset.custom_class(root=data_path, train=False, download=True,
                                    transform=dataset.transform_test, **custom_class_args)

        self.base_dataset = args.data_name
        self.train = train

        if self.train:
            if take_amount is not None:
                rng_state = np.random.get_state()
                np.random.seed(take_amount_seed)
                take_inds = np.random.choice(len(self.sup_indices),
                                             take_amount, replace=False)
                np.random.set_state(rng_state)

                logger = logging.getLogger()
                logger.info('Randomly taking only %d/%d examples from training'
                            ' set, seed=%d, indices=%s',
                            take_amount, len(self.sup_indices),
                            take_amount_seed, take_inds)
                self.targets = self.targets[take_inds]
                self.data = self.data[take_inds]

            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            if aux_data_filename is not None:
                aux_path = os.path.join(kwargs['root'], aux_data_filename)
                print("Loading data from %s" % aux_path)
                with open(aux_path, 'rb') as f:
                    aux = pickle.load(f)
                aux_data = aux['data']
                aux_targets = aux['extrapolated_targets']
                orig_len = len(self.data)

                if aux_take_amount != -1:
                    rng_state = np.random.get_state()
                    np.random.seed(take_amount_seed)
                    take_inds = np.random.choice(len(aux_data),
                                                 aux_take_amount, replace=False)
                    np.random.set_state(rng_state)

                    logger = logging.getLogger()
                    logger.info(
                        'Randomly taking only %d/%d examples from aux data'
                        ' set, seed=%d, indices=%s',
                        aux_take_amount, len(aux_data),
                        take_amount_seed, take_inds)
                    aux_data = aux_data[take_inds]
                    aux_targets = aux_targets[take_inds]

                self.data = np.concatenate((self.data, aux_data), axis=0)

                if not add_aux_labels:
                    self.targets.extend([-1] * len(aux_data))
                else:
                    self.targets.extend(aux_targets)
                # note that we use unsup indices to track the labeled datapoints
                # whose labels are "fake"
                self.unsup_indices.extend(
                    range(orig_len, orig_len+len(aux_data)))

            logger = logging.getLogger()
            logger.info("Training set")
            logger.info("Number of training samples: %d", len(self.targets))
            logger.info("Number of supervised samples: %d",
                        len(self.sup_indices))
            logger.info("Number of unsup samples: %d", len(self.unsup_indices))
            logger.info("Label (and pseudo-label) histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of training data: %s", np.shape(self.data))

        # Test set
        else:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            logger = logging.getLogger()
            logger.info("Test set")
            logger.info("Number of samples: %d", len(self.targets))
            logger.info("Label histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of data: %s", np.shape(self.data))

    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, value):
        self.dataset.targets = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        self.dataset.labels = self.targets  # because torchvision is annoying
        return self.dataset[item]

    def __repr__(self):
        fmt_str = 'Semisupervised Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.dataset.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.dataset.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.dataset.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class SemiSupervisedSampler(Sampler):
    """Balanced sampling from the labeled and unlabeled data"""
    def __init__(self, sup_inds, unsup_inds, batch_size, unsup_fraction=0.5,
                 num_batches=None):
        if unsup_fraction is None or unsup_fraction < 0:
            self.sup_inds = sup_inds + unsup_inds
            unsup_fraction = 0.0
        else:
            self.sup_inds = sup_inds
            self.unsup_inds = unsup_inds

        self.batch_size = batch_size
        unsup_batch_size = int(batch_size * unsup_fraction)
        self.sup_batch_size = batch_size - unsup_batch_size

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = int(
                np.ceil(len(self.sup_inds) / self.sup_batch_size))

        super().__init__(None)

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            sup_inds_shuffled = [self.sup_inds[i]
                                 for i in torch.randperm(len(self.sup_inds))]
            for sup_k in range(0, len(self.sup_inds), self.sup_batch_size):
                if batch_counter == self.num_batches:
                    break
                batch = sup_inds_shuffled[sup_k:(sup_k + self.sup_batch_size)]
                if self.sup_batch_size < self.batch_size:
                    batch.extend([self.unsup_inds[i] for i in
                                  torch.randint(high=len(self.unsup_inds),
                                                size=(
                                                    self.batch_size - len(
                                                        batch),),
                                                dtype=torch.int64)])
                # this shuffle operation is very important, without it
                # batch-norm / DataParallel hell ensues
                np.random.shuffle(batch)
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches
        
