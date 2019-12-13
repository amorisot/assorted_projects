import argparse
import csv
from collections import defaultdict
from collections import namedtuple
import datetime
import os
import subprocess
import json

from storage import stats_file_from_stats_folder


# Begin constants that depend on the folder structure
NAME_TO_FOLDER = {
    'jessica': '/home/s1164250/experiment_results/',
    'adrien': '/home/s1310324/experiment_results/',
    'michael': '/home/s1779598/experiment_results/',
}

# This is where our current experiments should end up
COMMITTED_RESULTS_FOLDER = 'experiment_analysis/CW4_results'
# TODO: Ugh, this assumes you're running in /scripts/runners
LOCAL_COMMITTED_RESULTS_FOLDER = os.path.join('../..', COMMITTED_RESULTS_FOLDER)

# This is where the GPU script puts things
EXPERIMENT_RESULTS_DEFAULT_PATH = os.path.join(os.environ['HOME'], 'experiment_results')

# CSV/JSON keys
EPOCH = 'epoch'
VAL_ACC_KEY = 'val_c_accuracy'
VAL_LOSS_KEY = 'val_c_loss'


# Command ids
CMD_RECENT = 'recent'
CMD_DETAILS = 'info'
CMD_LINKS = 'links'

# Whee, pretty terminal colors
# https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def set_color(text, color):
    return color + text + bcolors.ENDC


def argument_parser():
    parser = argparse.ArgumentParser(description='Helps manage experiment results files.')
    subparsers = parser.add_subparsers(dest='subparser_name')

    print_recent = subparsers.add_parser(CMD_RECENT, help='print recent runs')
    print_recent.add_argument(
        '--who',
        dest='who',
        help='whose experiment results to look at. one of {}'.format(list(NAME_TO_FOLDER.keys())))
    print_recent.add_argument(
        '--folder',
        dest='folder',
        default=EXPERIMENT_RESULTS_DEFAULT_PATH)
    print_recent.add_argument(
        '-n', '--number',
        dest='number',
        default=10,
        type=int,
        help='approx number to print')
    print_recent.add_argument('--commit-help', dest='commit_help', action='store_true',
        help=(
            'To commit the results, you must copy them to the experiment analysis folder.'
            'This command prints out the copy command. See also --stash'
        ))
    print_recent.add_argument('--stash', dest='stash', action='store_true',
        help=(
            'Like commit-help, but doesn\'t show run details',
        ))
    print_recent.add_argument('--stash-folder', dest='committed_results_folder', default=LOCAL_COMMITTED_RESULTS_FOLDER,
        help=(
            'Where to say to save the csv'
        ))

    print_details = subparsers.add_parser(CMD_DETAILS, help='given a CSV, pretty prints it')
    print_details.add_argument('--filename', dest='filename', required=True)

    print_links = subparsers.add_parser(CMD_LINKS, help='print markdown links for folder')

    return parser


def csv_lines_from_filename(filename):
    with open(filename) as f:
        lines = list(csv.DictReader(f))
    return lines


def json_lines_from_filename(filename):
    lines = []
    with open(filename) as f:
        for line in f:
            parsed = json.loads(line)
            # for now we can just check if 'epoch' is in the field to tell if it's logging an epoch
            if 'epoch' in parsed:
                lines.append(parsed)

    return lines


memoized_headers = {}
def json_headers_from_filename(filename):
    global memoized_headers

    if filename in memoized_headers:
        return memoized_headers[filename]

    lines = []
    with open(filename) as f:
        for line in f:
            parsed = json.loads(line)
            # for now we can just check if 'epoch' is in the field to tell if it's logging an epoch
            if 'epoch' not in parsed:
                lines.append(parsed)

    memoized_headers[filename] = lines

    return lines


memoized_filelines = {}
def lines_from_filename(filename):
    global memoized_filelines

    if filename in memoized_filelines:
        return memoized_filelines[filename]

    if filename.endswith('.csv'):
        lines = csv_lines_from_filename(filename)

    elif filename.endswith('.json'):
        lines = json_lines_from_filename(filename)

    memoized_filelines[filename] = lines

    return lines


def important_metric_type_from_header(header):
    return header.get('important_metric', {}).get('type', 'accuracy')


def print_results_from_file(results_filename):

    lines = lines_from_filename(results_filename)

    if not lines:
        print(set_color("No results in file", bcolors.FAIL))
        return

    json_header = json_headers_from_filename(results_filename)[0]
    # get the important metric type, defaulting to accuracy
    metric_type = important_metric_type_from_header(json_header)
    metric_key, get_best_metric = metric_key_from_important_metric_type(metric_type)

    accuracies = [
        line[metric_key]
        for line in lines
        if line[metric_key] is not None
    ]

    max_accuracy = get_best_metric(accuracies)

    headers = [
        'epoch',
        'run_id',
        'timestamp',
        'train_c_loss',
        'train_c_accuracy',
        'val_c_loss',
        'val_c_accuracy',
    ]

    print()
    print('\t'.join(
        line.replace('accuracy', 'acc').replace('run_id', 'run_id   ')  # hax
        for line in lines[0].keys()
    ))

    prev_timestamp = None
    for line in lines:
        if line[metric_key] == max_accuracy:
            color = bcolors.BOLD
            append = "\t*"
        else:
            color = bcolors.OKBLUE
            append = ""

        entries = []
        for k in headers:
            v = line[k]
            if k.endswith('accuracy'):
                v = '{:.03f}%  '.format(float(v) * 100)
            elif k.endswith('_loss'):
                v = '{:.08f}  '.format(float(v))
            elif k == 'timestamp':
                this_timestamp = int(v)
                if prev_timestamp is not None:
                    v = '{}   '.format(datetime.timedelta(seconds=(this_timestamp - prev_timestamp)))
                prev_timestamp = this_timestamp
            entries.append(str(v))

        line = '\t'.join(entries) + append
        print(set_color(line, color))


def description_of_run_id(run_id):
    # first make a pretty header
    timestamp = datetime.datetime.fromtimestamp(
        int(run_id)
    ).strftime('%Y-%m-%d %H:%M:%S')
    header = "Run {} from {}".format(
        set_color('r{}'.format(run_id), bcolors.OKBLUE),
        set_color(timestamp, bcolors.OKBLUE)
    )
    return header


def print_json_config_info(filename):
    headers = json_headers_from_filename(filename)
    if len(headers) > 1:
        print(set_color("This run continued from a previous run", bcolors.WARNING))
    # the most recent header is the first one
    header = headers[0]

    devices = device_from_filename(filename)

    print(
        'Device count: ' + set_color(devices, bcolors.BOLD)
        + '\t\tData: ' + set_color(str(header['data_provider_name']), bcolors.BOLD)
        + '\t\tBatch size: ' + set_color(str(header['batch_size']), bcolors.BOLD)
        + '\t\tEpochs: ' + set_color(str(header['epochs']), bcolors.BOLD)
        + '\t\tSha: ' + set_color(str(header.get('sha')), bcolors.BOLD)
    )


def maybe_run_id_from_filename(filename):
    run_id = None
    split_filename = filename.split('__')
    if len(split_filename) >= 2:
        # magic to extract from r123214324.csv
        run_id = split_filename[-1].split('.')[0][1:]

        # maybe remove the stats suffix
        run_id = run_id.rstrip('_stats')

        try:
            int(run_id)
        except ValueError:
            pass
    return run_id


def print_details(args):
    if not (args.filename.endswith('csv') or args.filename.endswith('json')):
        stats_filename = file_path_from_run_folder(args.filename)
    else:
        stats_filename = args.filename

    if not stats_filename:
        print(set_color('not an experiment run folder {}'.format(args.filename), bcolors.FAIL))
        return

    print(set_color('Pretty printing results for {}'.format(stats_filename), bcolors.HEADER))

    # if the run_id is readable, print the description based on that
    # TODO: when old configs are dropped, they will all have a run_id
    run_id = maybe_run_id_from_filename(stats_filename)
    if run_id is not None:
        print(description_of_run_id(run_id))

    # TODO: when all csvs are deprecated, can remove this
    if stats_filename.endswith('.json'):
        print_json_config_info(stats_filename)

    print_results_from_file(stats_filename)


def file_path_from_run_folder(folder):
    return stats_file_from_stats_folder(
        os.path.join(folder, 'stats'))


def device_from_filename(filename):
    devices = ' '
    if filename and filename.endswith('.json'):
        header = json_headers_from_filename(filename)[0]
        devices = '1cpu' if header['use_cpu'] else ''
        if not devices:
            devices = str(header.get('num_gpus') or header['num_devices']) + 'gpu'
    return devices


def get_other_runs_restored_from(filename):
    if not filename or not filename.endswith('.json'):
        return set()

    # look at this file's json headers and make a set with all the
    # non-None restore_from_locations, maybe trimming the last /
    return set(filter(None, (
        line['restore_from_location'].rstrip('/')
        for line in json_headers_from_filename(filename)
        if line.get('restore_from_location')
    )))

def print_recent(args):
    folder = args.folder
    if args.who:
        folder = NAME_TO_FOLDER[args.who]

    if not os.path.isdir(folder):
        print(set_color('{} not a folder, maybe set one with --folder?'.format(folder), bcolors.FAIL))
        return

    # maps run_id to config_id
    # run_ids aren't unique, so use a defaultdict to list
    exp_runs = defaultdict(list)

    # TODO: cache this and just check recently touched folders
    for config_id in os.listdir(folder):
        exp_path = os.path.join(folder, config_id)
        if not os.path.isdir(exp_path):
            continue
        for experiment_run in os.listdir(exp_path):
            try:
                int(experiment_run)
                exp_runs[experiment_run].append(config_id)
            except ValueError:
                print('skipping {} in {}, not a valid run_id'.format(
                    experiment_run,
                    exp_path))

    most_recent = sorted(exp_runs.keys(), reverse=True)[:args.number]

    runs_that_have_been_restored = set()

    for run_id in most_recent:
        for config_id in exp_runs[run_id]:
            folder_path = os.path.join(folder, config_id, run_id)
            filename = file_path_from_run_folder(folder_path)

            # This let's us check if a run has been continued
            runs_that_have_been_restored |= get_other_runs_restored_from(filename)
            restored_in_later_run = folder_path and folder_path in runs_that_have_been_restored

            if filename and filename.endswith('.json'):
                header = json_headers_from_filename(filename)[0]
            else:
                header = {}

            if not args.stash:
                print('{run_id_desc}  {early_stopped}\t{devices}\t{best_metric}\t{dataset}/{task}\t{path}'.format(
                    run_id_desc=description_of_run_id(run_id),
                    dataset=header.get('data_provider_name', '?'),
                    task=header.get('loss', {}).get('type', '?\t\t') + ('' if 'data_provider_name' in header else '\t'),
                    best_metric=pretty_max_important_metric_from_filename(filename),
                    early_stopped=check_if_early_stopped(filename, restored_in_later_run=restored_in_later_run),
                    devices=device_from_filename(filename),
                    path=folder_path,
                ))

            if (args.commit_help or args.stash) and filename is not None:
                copy_string = 'cp {} {}'.format(
                    filename,
                    LOCAL_COMMITTED_RESULTS_FOLDER
                )

                if os.path.isfile(os.path.join(args.committed_results_folder, os.path.basename(filename))):
                    print('# already saved {}'.format(os.path.basename(filename)))
                elif check_if_early_stopped(filename, symbols=False) != EARLY_STOPPED_STRING:
                    print('# not early stopped! {}'.format(copy_string))
                else:
                    print(copy_string)

    print('LEGEND: Run id and launch time; '
        'run status: (✓ means it early stopped, ? means it crashed or hasn\'t started,'
        ' ► means it\'s still going or timed out, ✘ means it ran all its epochs, + means we know it was restored from; '
        'max validation accuracy or lowest loss so far; '
        'device info; '
        'dataset and loss function name; '
        'run folder location.\n'
        'Using [filename] as the run folder location, '
        'to view results, run\n'
        '    python explore_experiment_results.py {} --filename [filename]\n'
        'to continue a job, run\n'
        '    sbatch restore_run_from_config_on_gpu.sh [filename]'
        ''.format(CMD_DETAILS))


def pretty_max_important_metric_from_filename(filename):
    max_important_metric = 'metric:   (X)     '
    if not filename:
        return max_important_metric

    if filename.endswith('.json'):
        json_header = json_headers_from_filename(filename)[0]

        # get the important metric type, defaulting to accuracy
        metric_type = important_metric_type_from_header(json_header)
    else:
        metric_type = 'accuracy'

    metric_key, get_best_metric = metric_key_from_important_metric_type(metric_type)

    lines = lines_from_filename(filename)
    if lines:
        v = get_best_metric(lines, key=lambda x: float(x[metric_key]))[metric_key]
        if metric_type == 'accuracy':
            max_important_metric = 'max acc:  ' + set_color(
                '{:.03f}%  '.format(float(v) * 100),
                bcolors.BOLD)
        elif metric_type == 'loss':
            max_important_metric = 'min loss: ' + set_color(
                '{:.03f}  '.format(float(v)),
                bcolors.BOLD)
    return max_important_metric


EARLY_STOPPED_STRING = 'early stopped'

def check_if_early_stopped(filename, symbols=True, restored_in_later_run=False):

    NO_LINES = '?' if symbols else '**error: no results**'
    EARLY_STOPPED = '✓' if symbols else 'early stopped'
    OUT_OF_EPOCHS = '✘' if symbols else '**ran for max epochs**'
    NOT_FINISHED = '►' if symbols else '**IN PROGRESS**'
    CONTINUED = '+' if symbols else '**continued**'

    if not filename:
        return False
    lines = lines_from_filename(filename)

    # if there are no lines at all, we either failed, or it hasn't started yet
    if not lines:
        return NO_LINES

    # now grab some info
    # first figure out which metric matters
    if filename.endswith('.json'):
        json_header = json_headers_from_filename(filename)[0]

        # get the important metric type, defaulting to accuracy
        metric_type = important_metric_type_from_header(json_header)
    else:
        metric_type = 'accuracy'

    metric_key, get_best_metric = metric_key_from_important_metric_type(metric_type)

    # first grab the metric with the highest epoch
    acc_epoch = int(
        get_best_metric(lines, key=lambda x: float(x[metric_key])
    )[EPOCH])
    # then grab the highest epoch
    highest_epoch = int(
        get_best_metric(lines, key=lambda x: int(x[EPOCH])
    )[EPOCH])

    target_epochs = None
    early_stop_after_n_epochs = 10
    if filename.endswith('.json'):
        header = json_headers_from_filename(filename)[0]
        target_epochs = header['epochs']
        early_stop_after_n_epochs = header['early_stop_after_n_epochs']

    # now check if it's ran long enough for early stopping.
    if acc_epoch < highest_epoch - early_stop_after_n_epochs:
        return EARLY_STOPPED

    if target_epochs:
        if highest_epoch + 1 >= header['epochs']:
            # we usually bump the max epochs and rerun if this happens
            return CONTINUED if restored_in_later_run else OUT_OF_EPOCHS

    return CONTINUED if restored_in_later_run else NOT_FINISHED


ConfigInfo = namedtuple('ConfigInfo', [
    'version',  # flags
    'config_id', 'run_id',  # usefuls stuff from all
    'sha', # useful stuff from new
])

def config_info_from_name(config_name, config_folder=LOCAL_COMMITTED_RESULTS_FOLDER):

    # set defaults
    version = 0
    sha = None

    split_name = config_name.split('__')

    if len(split_name) == 2:
        config_id, run_id = split_name
        run_id = run_id[:-len('.json')]

        if run_id.startswith('r'):
            run_id = run_id[1:]

        header = json_headers_from_filename(os.path.join(config_folder, config_name))[0]

        sha = header['sha']

        version = 2

    elif len(split_name) == 3:
        config_id, sha, run_id = split_name

        run_id = run_id[:-len('.csv')]

        if run_id.startswith('r'):
            run_id = run_id[1:]

        version = 1

    else:
        # an old style was just config_id_3424323

        split_name = config_name.split('_')
        config_id = '_'.join(split_name[:-1])
        run_id = split_name[-1]

    return ConfigInfo(
        version=version,
        config_id=config_id,
        sha=sha,
        run_id=run_id,
    )


def metric_key_from_important_metric_type(important_metric):
    # default is validation accuracy because that used to be all that was supported
    return {
        'accuracy': (VAL_ACC_KEY, max),
        'loss': (VAL_LOSS_KEY, min),
    }[important_metric]


def print_links(args):
    filenames = sorted(os.listdir(LOCAL_COMMITTED_RESULTS_FOLDER))

    link_prefix = "https://github.com/jessstringham/G63-MLP/blob"

    config_path = "scripts/runners/config"

    default_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()

    print('| config_id | config link | results | max val acc | status |')
    print('|--|--|--|--|--|')
    for filename in filenames:
        if not (filename.endswith('.csv') or filename.endswith('.json')):
            continue
        config_id = filename.split('__')[0].split('.')[0]

        config_info = config_info_from_name(filename)

        max_metric = pretty_max_important_metric_from_filename(
            os.path.join(LOCAL_COMMITTED_RESULTS_FOLDER, filename)
        )

        config_prefix = os.path.join(
            link_prefix,
            config_info.sha or default_sha,
            config_path
        )
        results_prefix = os.path.join(
            link_prefix,
            default_sha,
            COMMITTED_RESULTS_FOLDER
        )

        print('| {config_id} | [config]({config}) | [results]({results}) | {max_metric} | {is_finished} |'.format(
            config_id=config_info.config_id,
            config=os.path.join(config_prefix, config_info.config_id + '.yaml'),
            results=os.path.join(results_prefix, filename),
            max_metric=max_metric,
            is_finished=check_if_early_stopped(os.path.join(LOCAL_COMMITTED_RESULTS_FOLDER, filename), symbols=False)
        ))
        print()


COMMAND_TO_FUNC = {
    CMD_DETAILS: print_details,
    CMD_LINKS: print_links,
    CMD_RECENT: print_recent,
}


def main():
    args = argument_parser().parse_args()

    COMMAND_TO_FUNC[args.subparser_name](args)


if __name__ == '__main__':
    main()
