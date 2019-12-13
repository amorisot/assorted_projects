import argparse
import numpy as np
import tensorflow as tf

from data_loader_wrapper import datasets_from_provider_name
from network_config import parse_network_config
from layer_models import build_features
from tf_graph_helpers import count_parameters
from run_from_config import parameters_for_config_id
from run_from_config import parse_args


def argument_parser():
    parser = argparse.ArgumentParser(description='Run the requested network.')
    parser.add_argument('--network-config', dest='network_config',
                    help='Configuration that defines the network')
    return parser


def get_param_count(network_config):
    rng = np.random.RandomState(seed=network_config.seed)

    data_splits, input_placeholder, targets_placeholder = datasets_from_provider_name(rng, network_config)

    is_training_placeholder = tf.placeholder(tf.bool, name='training-flag')

    # produce predictions and get layer features to save for visual inspection
    layer_features, ongoing_result = build_features(
            network_config,
            input_placeholder,
            targets_placeholder,
            is_training_placeholder,
            data_splits,
            reuse=False,
        )
    losses_ops = network_config.loss.func(
            network_config,
            input_placeholder,
            targets_placeholder,
            is_training_placeholder,
            data_splits,
            layer_features,
            reuse=False,
        )

    return count_parameters(parameters_for_config_id(network_config))


def config_to_long_desc(network_config):

    def describe_layers(layers):
        result = []
        for layer in layers:

            dict_to_desc = {
                key: describer.get(key, lambda x: x)(value)
                for key, value in layer._asdict().items()
            }

            if type(layer).__name__ == 'Conv2DLayer':
                result.append((
                    '{kernel_length} conv. {filter_count} '
                    + ('{activation} units.' if layer.activation else 'filters.')
                    + ' {strides}'
                ).format(**dict_to_desc))
            elif type(layer).__name__ == 'MaxPoolLayer':
                result.append((
                    'maxpooling {pool_size}'
                ).format(**dict_to_desc))
            elif type(layer).__name__ == 'DenseLayer':
                result.append((
                    'FC '
                    + '{units}'
                    + ' {activation} units'
                ).format(**dict_to_desc))
            elif type(layer).__name__ == 'SoftmaxPrediction':
                result.append('softmax')
            elif type(layer).__name__ == 'PrimaryCapsule':
                # just skip argmax
                result.append((
                    '{kernel_length} primary capsule. '
                    + '{caps_count} capsules '
                    + '{caps_dim}; '
                    + '{strides}; '
                    + '{activation}'
                ).format(**dict_to_desc))
            elif type(layer).__name__ == 'OutCapsule':
                # just skip argmax
                result.append((
                    'digit capsule: '
                    + '{caps_count} capsules of dim '
                    + '{caps_dim}.'
                ).format(**dict_to_desc))
            elif type(layer).__name__ == 'DynamicRouting':
                # just skip argmax
                result.append((
                    'dynamic routing:  '
                    + '{iterations}.'
                ).format(**dict_to_desc))
            elif type(layer).__name__ == 'CapsulePrediction':
                result.append('(prediction)')
            elif type(layer).__name__ == 'CapsuleMasking':
                result.append('mask')
            elif type(layer).__name__ == 'BatchNormLayer':
                result.append('batch norm')
            elif type(layer).__name__ == 'ArgMaxLayer':
                # just skip argmax
                pass
            elif type(layer).__name__ == 'ReshapeLayer':
                # just imply reshaping
                pass

            else:
                raise Exception("I don't know how to draw layers of type {}".format(type(layer).__name__))

        return result

    desc_data_provider = {
        'mnist': '\\textit{MNIST}: Input $28 \\times 28 \\times 1$, 10 classes',
        'emnist': '\\textit{EMNIST}: Input $28 \\times 28 \\times 1$, 47 classes',
        'cifar10': '\\textit{CIFAR10}: Input $32 \\times 32 \\times 3$, 10 classes',
        'smallnorb': '\\textit{small NORB}: Input $48 \\times 48 \\times 1$, 5 classes',
        'tinysmallnorb': '\\textit{small NORB}: Input $28 \\times 28 \\times 1$, 5 classes',
        #'smallnorb': '\\textit{small NORB}: Input $48 \\times 48 \\times 1$, 5 classes',
    }

    desc_activation = {
        'leaky_relu': 'Leaky ReLU',
        'relu': 'ReLU',
        'sigmoid': 'Sigmoid',
    }

    describer = {
        'data_provider_name': lambda x: desc_data_provider[x],
        'batch_size': lambda x: 'batchsize {x}'.format(x=x),
        'layers': lambda x: '\\\\ \hline \n'.join(describe_layers(x)),
        'strides': lambda x: 'stride {x}'.format(x=x),
        'kernel_length': lambda x: '${x} \\times {x}$'.format(x=x),
        'shape': lambda x: '$' + ' \\times '.join(map(str, x)) + '$',
        'activation': lambda x: desc_activation[x],
        'pool_size': lambda x: '$' + ' \\times '.join(map(str, x)) + '$',
        #'caps_count': lambda x: 'count: {x}'.format(x=x),
        'dim_count': lambda x: '{x}-dim'.format(x=x),
        'iterations': lambda x: '{x} iterations'.format(x=x),
    }

    dict_to_desc = {
        key: describer.get(key, lambda x: x)(value)
        for key, value in network_config._asdict().items()
    }

    table_prefix = '''
\\begin{center}
  \\begin{tabular}{ | c |}
    \\hline'''

    table_suffix = '''\\\\ \\hline
  \\end{tabular}
\\end{center}
'''

    table_prefix = '''
\\begin{table}[tb]
\\vskip 3mm
\\begin{center}
\\begin{small}
\\begin{sc}
\\begin{tabular}{| c |}
\\hline'''

    table_suffix = '''\\\\ \\hline
\\end{tabular}
\\end{sc}
\\end{small}
\\caption{TODO}
\\label{tab:arch}
\\end{center}
\\vskip -3mm
\\end{table}
'''

    #dict_to_desc['parameters'] = '{:0.1f}M'.format(get_param_count(network_config) / 1e6)
    dict_to_desc['parameters'] = ''

    dict_to_desc['config_id'] = network_config.config_id

    return '''
% {config_id}
{table_prefix}
\\abovespace\\belowspace
{config_id} \\\\ \\hline
{data_provider_name}; {batch_size}\\\\ \\hline
{layers} \\\\ \\hline \\hline
Parameter count: {parameters}
{table_suffix}


    '''.format(
        table_prefix=table_prefix,
        table_suffix=table_suffix,
        **dict_to_desc
    )


def main():
    args = argument_parser().parse_args()
    fake_args = parse_args(['--network-config', args.network_config, '--debug'])
    network_config = parse_network_config(args.network_config, fake_args)

    print(config_to_long_desc(network_config))


if __name__ == '__main__':
    main()
