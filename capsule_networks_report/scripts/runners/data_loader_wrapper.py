from collections import namedtuple

import tensorflow as tf

from modified_data_providers import CIFAR10DataProvider
from modified_data_providers import EMNISTDataProvider
from modified_data_providers import MNISTDataProvider
from modified_data_providers import SmallNORBDataProvider
from modified_data_providers import MTLSmallNORBDataProvider
from modified_data_providers import TinySmallNORBDataProvider
from modified_data_providers import MTLTinySmallNORBDataProvider

DataSplits = namedtuple('DataSplits', [
    'train_data',
    'val_data',
    'test_data',
])

DatasetConfig = namedtuple('DatasetConfig', [
    'provider_class',  # data_provider classname
    'input_data_type',  # tf data type for the input
    'input_shape',  # suffix of input shape, i.e. shape of one example
    'output_data_type',  # tf data type for the output
    'output_shape',  # suffix of output shape, i.e. shape of one target
    'dataset_specific_kwargs',  # dictionary for kwargs for dataset
])


DATASET_NAME_TO_DATASET_CONFIG = {
    'cifar10': {
        'provider_class': CIFAR10DataProvider,
        'input_shape': [32, 32, 3],
    },
    'mnist': {
        'provider_class': MNISTDataProvider,
        'input_shape': [28, 28, 1],
    },
    'emnist': {
        'provider_class': EMNISTDataProvider,
        'input_shape': [28, 28, 1],
    },
    'smallnorb': {
        'provider_class': SmallNORBDataProvider,
        'input_shape': [48, 48, 1],
    },
    'mtlsmallnorb': {
        'provider_class': MTLSmallNORBDataProvider,
        'input_shape': [48, 48, 1],
        'output_shape': [2]
    },
    'tinysmallnorb': {
        'provider_class': TinySmallNORBDataProvider,
        'input_shape': [28, 28, 1],
    },
    'mtltinysmallnorb': {
        'provider_class': MTLTinySmallNORBDataProvider,
        'input_shape': [28, 28, 1],
        'output_shape': [2]
    },
}

def datasets_from_provider_name(rng, network_config):
    dataset_config_dict = {
        'input_data_type': tf.float32,
        'output_shape': [],
        'output_data_type': tf.int64,
        'dataset_specific_kwargs': {},
    }

    dataset_config_dict.update(DATASET_NAME_TO_DATASET_CONFIG[network_config.data_provider_name])

    dataset_config = DatasetConfig(**dataset_config_dict)

    input_placeholder = tf.placeholder(
        dataset_config.input_data_type,
        [network_config.num_devices, network_config.batch_size] + dataset_config.input_shape,
        'data-inputs'
    )
    targets_placeholder = tf.placeholder(
        dataset_config.output_data_type,
        [network_config.num_devices, network_config.batch_size] + dataset_config.output_shape,
        'data-targets'
    )

    return DataSplits(
        train_data=dataset_config.provider_class(
            which_set="train",
            batch_size=network_config.batch_size,
            num_devices=network_config.num_devices,
            rng=rng,
            random_sampling=True,
            **dataset_config.dataset_specific_kwargs),
        val_data=dataset_config.provider_class(
            which_set="valid",
            batch_size=network_config.batch_size,
            num_devices=network_config.num_devices,
            rng=rng,
            **dataset_config.dataset_specific_kwargs),
        test_data = dataset_config.provider_class(
            which_set="test",
            batch_size=network_config.batch_size,
            num_devices=network_config.num_devices,
            rng=rng,
            **dataset_config.dataset_specific_kwargs),
    ), input_placeholder, targets_placeholder

