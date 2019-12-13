'''This crawls in all yaml files, and deletes the first element of reshape'''

from yaml_configurator import run_on_all_yamls_in_folder_skeleton


def remove_salt(yaml, dest):

	for layer in yaml['layers']:
		if layer['type'] == 'reshape':
			if not (layer['shape'][0] == 50 or layer['shape'][0] == -1):
				print('reshape doesn\'t start with -1 or 50')
				continue
			print('updating {} to {}'.format(layer['shape'], layer['shape'][1:]))
			layer['shape'] = layer['shape'][1:]

	return yaml, dest

def main():
	run_on_all_yamls_in_folder_skeleton(remove_salt)


if __name__ == '__main__':
	main()
