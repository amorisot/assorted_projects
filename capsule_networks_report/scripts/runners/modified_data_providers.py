# -*- coding: utf-8 -*-
"""Data providers.
Copied from https://raw.githubusercontent.com/CSTR-Edinburgh/mlpractical/mlp2017-8/semester_2_materials/data_providers.py

This module provides classes for loading datasets and iterating over batches of
data points.
"""

import os

import numpy as np
DEFAULT_SEED = 22012018


# Sorry. Change this to True if you want to use GPU on smallnorb
USE_SMALLNORB_GPU_FOLDER_LOCATION = False

# You probably don't need to change this, just change USE_SMALLNORB_GPU_FOLDER_LOCATION
SMALLNORB_LOCATION = '/home/s1164250/mlpractical/data/' if USE_SMALLNORB_GPU_FOLDER_LOCATION else os.environ['MLP_DATA_DIR']

class DataProvider(object):
    """Generic data provider."""

    def __init__(
        self,
        inputs,
        targets,
        batch_size,
        num_devices,
        random_sampling=True,
        rng=None,
    ):
        """Create a new data provider object.

        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            random_sampling (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.inputs = inputs
        self.targets = targets

        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')

        self._batch_size = batch_size

        self.num_devices = num_devices
        self.num_batches = self.inputs.shape[0] // (self.batch_size * self.num_devices)
        self.random_sampling = random_sampling
        self._current_order = np.arange(inputs.shape[0])

        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)

        self.rng = rng
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size

    def __iter__(self):
        """Implements Python iterator interface.

        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data)"""
        self._curr_batch = 0

    def __next__(self):
        return self.next()

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        example_count = size=self.batch_size * self.num_devices
        if self.random_sampling:
            batch_slice = self.rng.choice(self.inputs.shape[0], size=example_count, replace=False)

        else:
            batch_slice = slice(self._curr_batch * example_count,
                                (self._curr_batch + 1) * example_count)

        inputs_batch = self.inputs[batch_slice].reshape(
            self.num_devices,
            self.batch_size,
            *self.inputs.shape[1:]
        )
        targets_batch = self.targets[batch_slice].reshape(self.num_devices, self.batch_size, *self.targets.shape[1:])
        self._curr_batch += 1
        return inputs_batch, targets_batch

class MNISTDataProvider(DataProvider):
    """Data provider for MNIST handwritten digit images."""

    def __init__(
        self,
        which_set,
        batch_size,
        num_devices,
        random_sampling=True,
        rng=None,
        flatten=False,
    ):
        """Create a new MNIST data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'eval'. Determines which
                portion of the MNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            random_sampling (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 10
        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # MLP_DATA_DIR environment variable should point to the data directory
        data_path = os.path.join(
            os.environ['MLP_DATA_DIR'], 'mnist-{0}.npz'.format(which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load data from compressed numpy file
        loaded = np.load(data_path)
        inputs, targets = loaded['inputs'], loaded['targets']
        if flatten:
            inputs = np.reshape(inputs, newshape=(-1, 28*28))
        else:
            inputs = np.reshape(inputs, newshape=(-1, 28, 28, 1))
        inputs = inputs.astype(np.float32)
        # pass the loaded data to the parent class __init__
        super(MNISTDataProvider, self).__init__(
            inputs=inputs,
            targets=targets,
            batch_size=batch_size,
            num_devices=num_devices,
            random_sampling=random_sampling,
            rng=rng,
        )

class EMNISTDataProvider(DataProvider):
    """Data provider for EMNIST handwritten digit images."""

    def __init__(
        self,
        which_set,
        batch_size,
        num_devices,
        random_sampling=True,
        rng=None,
        flatten=False,
    ):
        """Create a new EMNIST data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'eval'. Determines which
                portion of the EMNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            random_sampling (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 47
        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # MLP_DATA_DIR environment variable should point to the data directory
        data_path = os.path.join(
            os.environ['MLP_DATA_DIR'], 'emnist-{0}.npz'.format(which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load data from compressed numpy file
        loaded = np.load(data_path)

        inputs, targets = loaded['inputs'], loaded['targets']
        inputs = inputs.astype(np.float32)
        if flatten:
            inputs = np.reshape(inputs, newshape=(-1, 28*28))
        else:
            inputs = np.expand_dims(inputs, axis=3)
        inputs = inputs / 255.0

        # pass the loaded data to the parent class __init__
        super(EMNISTDataProvider, self).__init__(
            inputs=inputs,
            targets=targets,
            batch_size=batch_size,
            num_devices=num_devices,
            random_sampling=random_sampling,
            rng=rng,
        )


class CIFAR10DataProvider(DataProvider):
    """Data provider for CIFAR-10 object images."""

    def __init__(
        self,
        which_set,
        batch_size,
        num_devices,
        random_sampling=True,
        rng=None,
        flatten=False,
    ):
        """Create a new EMNIST data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'eval'. Determines which
                portion of the EMNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            random_sampling (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 10
        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # MLP_DATA_DIR environment variable should point to the data directory
        data_path = os.path.join(
            os.environ['MLP_DATA_DIR'], 'cifar10-{0}.npz'.format(which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load data from compressed numpy file
        loaded = np.load(data_path)

        inputs, targets = loaded['inputs'], loaded['targets']
        inputs = inputs.astype(np.float32)
        if flatten:
            inputs = np.reshape(inputs, newshape=(-1, 32*32*3))
        else:
            inputs = np.reshape(inputs, newshape=(-1, 3, 32, 32))
            inputs = np.transpose(inputs, axes=(0, 2, 3, 1))

        inputs = inputs / 255.0
        # label map gives strings corresponding to integer label targets


        # pass the loaded data to the parent class __init__
        super(CIFAR10DataProvider, self).__init__(
            inputs=inputs,
            targets=targets,
            batch_size=batch_size,
            num_devices=num_devices,
            random_sampling=random_sampling,
            rng=rng,
        )


def prep_smallnorb_inputs(inputs, downsample, flatten, scale_down_factor=2):
    inputs = inputs.astype(np.float32)
    inputs = inputs[:, 0, :, :] # limit to camera 1

    width = 96

    if downsample:
        inputs = inputs[:, ::scale_down_factor, ::scale_down_factor]
        width //= scale_down_factor

    if flatten:
        inputs = np.reshape(inputs, newshape=(-1, width * width))
    else:
        inputs = np.reshape(inputs, newshape=(-1, 1, width, width))
        inputs = np.transpose(inputs, axes=(0, 2, 3, 1))
    inputs = inputs / 255.0

    return inputs

class SmallNORBDataProvider(DataProvider):
    """Data provider for SmallNORB object images."""

    def __init__(
        self,
        which_set,
        batch_size,
        num_devices,
        random_sampling=True,
        rng=None,
        flatten=False,
        downsample=True,
    ):
        """Create a new SmallNORB data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'eval'. Determines which
                portion of the SmallNORB data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            random_sampling (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 5
        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # MLP_DATA_DIR environment variable should point to the data directory
        data_path = os.path.join(
            SMALLNORB_LOCATION, 'smallnorb20180213-{0}.npz'.format(which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load data from compressed numpy file
        loaded = np.load(data_path)

        raw_inputs, targets = loaded['inputs'], loaded['targets']
        inputs = prep_smallnorb_inputs(raw_inputs, downsample, flatten)

        # pass the loaded data to the parent class __init__
        super(SmallNORBDataProvider, self).__init__(
            inputs=inputs,
            targets=targets,
            batch_size=batch_size,
            num_devices=num_devices,
            random_sampling=random_sampling,
            rng=rng,
        )


class TinySmallNORBDataProvider(DataProvider):
    """Data provider for SmallNORB object images."""

    def __init__(
        self,
        which_set,
        batch_size,
        num_devices,
        random_sampling=True,
        rng=None,
        flatten=False,
        downsample=True,
    ):
        """Create a new SmallNORB data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'eval'. Determines which
                portion of the SmallNORB data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            random_sampling (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 5
        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # MLP_DATA_DIR environment variable should point to the data directory
        data_path = os.path.join(
            SMALLNORB_LOCATION, 'smallnorb20180213-{0}.npz'.format(which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load data from compressed numpy file
        loaded = np.load(data_path)

        raw_inputs, targets = loaded['inputs'], loaded['targets']
        inputs = prep_smallnorb_inputs(raw_inputs, downsample, flatten, scale_down_factor=3)
        inputs = inputs[:, 2:30, 2:30]

        # pass the loaded data to the parent class __init__
        super(TinySmallNORBDataProvider, self).__init__(
            inputs=inputs,
            targets=targets,
            batch_size=batch_size,
            num_devices=num_devices,
            random_sampling=random_sampling,
            rng=rng,
        )


class MTLSmallNORBDataProvider(DataProvider):
    """Data provider for SmallNORB object images and multi-learning targets"""

    def __init__(
        self,
        which_set,
        batch_size,
        num_devices,
        random_sampling=True,
        rng=None,
        flatten=False,
        downsample=True,
    ):
        """Create a new SmallNORB data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'eval'. Determines which
                portion of the SmallNORB data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            random_sampling (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 5
        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # MLP_DATA_DIR environment variable should point to the data directory
        data_path = os.path.join(
            SMALLNORB_LOCATION, 'smallnorb-mtl-20180309-{0}.npz'.format(which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load data from compressed numpy file
        loaded = np.load(data_path)

        raw_inputs, targets = loaded['inputs'], loaded['targets']
        inputs = prep_smallnorb_inputs(raw_inputs, downsample, flatten)

        # pass the loaded data to the parent class __init__
        super(MTLSmallNORBDataProvider, self).__init__(
            inputs=inputs,
            targets=targets,
            batch_size=batch_size,
            num_devices=num_devices,
            random_sampling=random_sampling,
            rng=rng,
        )

class MTLTinySmallNORBDataProvider(DataProvider):
    """Data provider for SmallNORB object images and multi-learning targets"""

    def __init__(
        self,
        which_set,
        batch_size,
        num_devices,
        random_sampling=True,
        rng=None,
        flatten=False,
        downsample=True,
    ):
        """Create a new SmallNORB data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'eval'. Determines which
                portion of the SmallNORB data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            random_sampling (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 5
        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # MLP_DATA_DIR environment variable should point to the data directory
        data_path = os.path.join(
            SMALLNORB_LOCATION, 'smallnorb-mtl-20180309-{0}.npz'.format(which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load data from compressed numpy file
        loaded = np.load(data_path)

        raw_inputs, targets = loaded['inputs'], loaded['targets']
        inputs = prep_smallnorb_inputs(raw_inputs, downsample, flatten, scale_down_factor=3)
        inputs = inputs[:, 2:30, 2:30]

        # pass the loaded data to the parent class __init__
        super(MTLTinySmallNORBDataProvider, self).__init__(
            inputs=inputs,
            targets=targets,
            batch_size=batch_size,
            num_devices=num_devices,
            random_sampling=random_sampling,
            rng=rng,
        )
