'''This crawls in all yaml files, and deletes the salt line'''

from yaml_configurator import run_on_all_yamls_in_folder_skeleton


def lint(yaml, dest):
	return yaml, dest

def main():
	run_on_all_yamls_in_folder_skeleton(lint)


if __name__ == '__main__':
	main()
