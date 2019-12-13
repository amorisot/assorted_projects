'''This crawls in all yaml files, and deletes the salt line'''

from yaml_configurator import run_on_all_yamls_in_folder_skeleton


def remove_salt(yaml, dest):
	result = yaml.pop('seed', None)
	if result is not None:
		print('removed!')
	return yaml, dest

def main():
	run_on_all_yamls_in_folder_skeleton(remove_salt)


if __name__ == '__main__':
	main()
