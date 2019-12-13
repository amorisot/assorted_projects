'''This file helps convert the yaml into the NetworkConfig object, which is used to
build the network. The main entrypoint is NetworkConfig.parse_config(config_location),
where config_location is the filepath for the config.

To add new fields to the top-level configuration, e.g. for things like "number of epochs",
add here. For things specific to layers, see layer_models.
'''
from collections import namedtuple
import time
import os.path
import json
import subprocess
import yaml

from layer_models import layers_from_list
from layer_models import layer_idx_with_id
from loss_functions import build_caps_margin_losses_ops
from loss_functions import build_softmax_crossentropy_losses_op
from loss_functions import LOSS_TYPE_TO_FUNC_AND_INFO_TUPLE
from data_augmentation import DATA_AUGMENTATION_FUNC_FROM_NAME
from data_augmentation import DEFAULT_DATA_AUGMENTATION_NAME


def get_code_sha():
    return subprocess.check_output(['git', 'rev-parse', '--short=7', 'HEAD']).decode('utf-8').strip()


OptimizerKwargs = namedtuple('OptimizerKwargs', ['learning_rate', 'beta1'])

def optimizer_info_from_dict(raw_optimizer_dict_kwargs):
    if raw_optimizer_dict_kwargs is None:
        raw_optimizer_dict_kwargs = {}

    # defaults
    optimizer_dict_kwargs = {
        'learning_rate': 1e-3,
        'beta1': 0.9,
    }

    optimizer_dict_kwargs.update(raw_optimizer_dict_kwargs)

    return OptimizerKwargs(**optimizer_dict_kwargs)


def config_location_to_config_id(config_location):
    base = os.path.basename(config_location)
    return base.split('.')[0]


def loss_info_from_dict(loss_dict, layers):
    loss_type = loss_dict.pop('type')
    loss_func, tuple_type, loss_info_dict = LOSS_TYPE_TO_FUNC_AND_INFO_TUPLE[loss_type]

    loss_info_dict['func'] = loss_func

    for key, value in loss_dict.items():
        if key.endswith('_id'):
            loss_info_dict[key] = layer_idx_with_id(layers, value)
        else:
            loss_info_dict[key] = value

    return tuple_type(**loss_info_dict)


# Important Metrics are used to do early stopping and to report.
# Right now, the namedtuple type is what determines what early stops
# (see important_metrics.py).
AccuracyMetric = namedtuple('AccuracyMetric', ['prediction_layer_idx'])
LossMetric = namedtuple('LossMetric', ['prediction_layer_idx'])

def parse_important_metric(important_metric_dict, layers):
    metric_type = important_metric_dict.pop('type')

    if metric_type == 'accuracy':
        return AccuracyMetric(
            prediction_layer_idx=layer_idx_with_id(
                layers,
                important_metric_dict['prediction_id'],
            )
        )
    elif metric_type == 'loss':
        prediction_layer_idx = None
        if 'prediction_id' in important_metric_dict:
            prediction_layer_idx = layer_idx_with_id(
                layers,
                important_metric_dict['prediction_id'],
            )
        return LossMetric(prediction_layer_idx)
    else:
        raise Exception('Unknown metric type `{}`'.format(metric_type))


NetworkConfig = namedtuple('NetworkConfig', [
    'config_id',  # filename of this config
    'data_provider_name',  # dataset to use
    'data_augmentation_func',  # which scheme to use to augment data
    'batch_size',  # size of minibatches
    'seed',  # random number generator seed
    'loss',
    'epochs',  # number of epochs to run for
    'layers',  # list of layer namedtuple
    'early_stop_after_n_epochs', # int of how many steps to have worse results before quitting
    'important_metric',  # important metric that should be used for early stopping
    'optimizer_kwargs',  # information about the optimizer
    'use_cpu',  # if should use one cpu
    'num_devices', # how many devices to use.
    'serialized',  # dictionary representing this
    'restore_from_location',  # where this run was restored from
])


def _network_config_parse_and_validate(config_dict):
    new_config_dict = {}

    new_config_dict['data_provider_name'] = config_dict['data_provider_name']

    new_config_dict['data_augmentation_func'] = DATA_AUGMENTATION_FUNC_FROM_NAME[
        config_dict.get('data_augmentation_name', DEFAULT_DATA_AUGMENTATION_NAME)
    ]

    new_config_dict['batch_size'] = int(config_dict['batch_size'])

    new_config_dict['epochs'] = int(config_dict['epochs'])

    new_config_dict['early_stop_after_n_epochs'] = int(config_dict['early_stop_after_n_epochs'])

    new_config_dict['layers'] = layers_from_list(config_dict['layers'])

    new_config_dict['optimizer_kwargs'] = optimizer_info_from_dict(
        config_dict.get('optimizer_kwargs')
    )

    # Things that need to happen after parsing the layers
    # TODO: replace this with the new prediction dict and remove special case stuff.
    if 'prediction_id' in config_dict:
        new_config_dict['important_metric'] = AccuracyMetric(
            prediction_layer_idx=layer_idx_with_id(
                new_config_dict['layers'],
                config_dict['prediction_id'],
            )
        )
    else:
        assert 'prediction_id' not in config_dict
        new_config_dict['important_metric'] = parse_important_metric(
            config_dict['important_metric'],
            new_config_dict['layers'],
        )

    new_config_dict['loss'] = loss_info_from_dict(config_dict['loss'], new_config_dict['layers'])

    return new_config_dict


def _network_config_parse_args(args):
    new_config_dict = {}

    new_config_dict['use_cpu'] = (args.num_gpus == 0)

    new_config_dict['num_devices'] = max(args.num_gpus, 1)
    assert new_config_dict['num_devices'] >= 1
    assert new_config_dict['num_devices'] <= 8  # arbitrary upper limit

    new_config_dict['seed'] = args.seed

    new_config_dict['restore_from_location'] = args.restore_from_location

    return new_config_dict


def parse_network_config(config_location, args):
    with open(config_location) as f:
        config_dict = yaml.load(f)

    # use the dict from _network_config_parse_args for representing args
    # use the raw config_dict for representing the config
    serialized_dict = _network_config_parse_args(args)
    serialized_dict.update(config_dict)
    serialized_dict['sha'] = get_code_sha()
    serialized = json.dumps(serialized_dict)

    return NetworkConfig(
        config_id=config_location_to_config_id(config_location),
        serialized=serialized,
        **_network_config_parse_args(args),
        **_network_config_parse_and_validate(config_dict)
    )

