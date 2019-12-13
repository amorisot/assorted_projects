import argparse
import os

import ruamel.yaml

def run_on_yaml_with_parser(yaml_func, args_func=None):
    parser = argparse.ArgumentParser(description='Modify the requested yaml file.')
    parser.add_argument('network_config', help='Configuration that defines the network')
    parser.add_argument('--debug', dest='is_debug', action='store_true',
                    help='Don\'t modify, just print results')

    if args_func is not None:
        parser = args_func(parser)

    args = parser.parse_args()

    yaml = args.network_config

    print('Processing {}'.format(yaml))

    yaml_dict = parse_network_config_to_dict(yaml)
    new_yaml_dict, dest = yaml_func(yaml_dict, yaml, args)

    print('Output {}'.format(dest))

    save_network_config(new_yaml_dict, dest, args.is_debug, should_print=args.is_debug)


def all_yamls_from_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.yaml'):
                yield os.path.join(root, file)

def run_on_all_yamls_in_folder_skeleton(func):
    parser = argparse.ArgumentParser()
    parser.add_argument('folder',
                    help='folder that contains the yamls')
    parser.add_argument('--debug', dest='is_debug', action='store_true',
                    help='Don\'t modify, just print results')

    args = parser.parse_args()

    for yaml in all_yamls_from_folder(args.folder):
        print('Processing {}'.format(yaml))

        yaml_dict = parse_network_config_to_dict(yaml)
        new_yaml_dict, dest = func(yaml_dict, yaml)

        save_network_config(new_yaml_dict, dest, args.is_debug, should_print=False)


def parse_network_config_to_dict(config_location):
    with open(config_location) as f:
        config_dict = ruamel.yaml.load(f, ruamel.yaml.RoundTripLoader)

    return config_dict

def save_network_config(network_config, output_location, is_debug=False, should_print=False):
    if should_print:
        print(ruamel.yaml.dump(network_config, Dumper=ruamel.yaml.RoundTripDumper))

    if is_debug:
        return

    else:
        with open(output_location, 'w') as f:
            f.write(ruamel.yaml.dump(network_config, Dumper=ruamel.yaml.RoundTripDumper))


