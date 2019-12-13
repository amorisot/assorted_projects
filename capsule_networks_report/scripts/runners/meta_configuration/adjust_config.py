'''This is meant to be a general purpose config adjuster. It reads in the
config, makes the designated change, and saves it in an appropriately-named
file.'''

from yaml_configurator import run_on_yaml_with_parser

def clean_for_filename(value):
	return str(value).replace('.', '_')

def update_filename(source, field, value):
	if field in source:
		raise Exception("{} already has {} and I don't know how to do this yet! update the script to add this feature".format(source, field))

	basename = source[:-len('.yaml')]

	basename = '{}_{}_{}'.format(basename, field, clean_for_filename(value))

	return basename + '.yaml'


def add_arguments(parser):
	parser.add_argument('--learning-rate', dest='learning_rate', type=float)
	return parser


def adjust_learning_rate(yaml, source, value):
	print("Adjusting learning rate to {}".format(value))

	yaml.update({
		'optimizer_kwargs': {
		    'learning_rate': value,
		}
	})
	dest = update_filename(source, 'learning_rate', value)
	return yaml, dest


def adjust_config(yaml, source, args):
	if args.learning_rate is not None:
		yaml, dest = adjust_learning_rate(yaml, source, args.learning_rate)


	return yaml, dest

def main():
	run_on_yaml_with_parser(adjust_config, add_arguments)


if __name__ == '__main__':
	main()
