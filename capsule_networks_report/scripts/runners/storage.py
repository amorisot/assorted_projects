import csv
import os
import time
import json


def config_id_from_location(stats_folder):
    if stats_folder.endswith('/'):
        stats_folder = stats_folder[:-1]
    return stats_folder.split('/')[-2]


def stats_file_from_stats_folder(stats_folder):
    '''Find the stats file located in this folder.
    There should be only one csv, so this just finds it.'''

    if not os.path.isdir(stats_folder):
        return

    for file in os.listdir(stats_folder):
        if file.endswith('_stats.json'):
            return os.path.join(stats_folder, file)

        if file.endswith('.csv'):
            return os.path.join(stats_folder, file)



def mkdir_p(path, noop, force_unique=False):
    if noop:
        return
    if force_unique:
        try:
            os.makedirs(path)
        except FileExistsError:
            raise Exception('Oh no we tried to assign the same run_id to two jobs. Sorry, just rerun this.')
    elif not os.path.exists(path):
        os.makedirs(path)



class FolderPicker():

    BEST_VAL_FOLDERNAME = 'best_val_model'

    def __init__(self, output_location, network_config, run_id, mkdirs=False, restorer=False):
        # mkdirs will create folder
        # restorer will lookup the label based on run_id
        #
        # folder structure is
        # /output_location
        #   /config_id
        #    /run_id
        #      /checkpoints
        #        /final_trained.ckpt
        #        /best_val_model-#  (be sure to pass in a global step)
        #      /stats
        #      /tf_logdir
        mkdir_is_noop = not mkdirs

        # make the function a no-op if we shouldn't make folders
        config_id_location = os.path.join(output_location, network_config.config_id)
        mkdir_p(config_id_location, mkdir_is_noop)

        # avoid a race condition by failing if this folder exists
        run_id_location = os.path.join(config_id_location, run_id)
        mkdir_p(run_id_location, mkdir_is_noop, force_unique=True)

        checkpoints_location = os.path.join(run_id_location, 'checkpoints')
        mkdir_p(checkpoints_location, mkdir_is_noop)

        stats_location = os.path.join(run_id_location, 'stats')
        mkdir_p(stats_location, mkdir_is_noop)

        self.config_id_location = config_id_location
        self.run_id = run_id
        self.run_id_location = run_id_location
        self.checkpoints_location = checkpoints_location
        self.stats_location = stats_location

        # if this folder picker is helping us find old folders, then
        # search for the stats file to use as the "label".
        # Otherwise, create a new label.
        if restorer:
            stat_file = stats_file_from_stats_folder(self.stats_location)
            if not stat_file:
                raise Exception('no stats file in folder {}'.format(self.stats_location))

            if stat_file.endswith('_stats.json'):
                self.label = stat_file[:-len('_stats.json')]
            else:
                raise Exception('sorry need to start from a more recent run!')
        else:
            self.label = '{config_id}__r{run_id}'.format(
                config_id=network_config.config_id,
                run_id=self.run_id,
            )

        self.restorer = restorer

    @property
    def path_for_trained_model(self):
        return os.path.join(self.checkpoints_location, 'final_trained.ckpt')

    @property
    def path_for_epoch_stats(self):
        return os.path.join(self.stats_location, '{}_stats.json'.format(self.label))

    @property
    def path_for_tf_logs_dir(self):
        return os.path.join(self.run_id_location, 'tf_logs')

    @property
    def path_for_best_val_model(self):
        # Don't forget to use global_step with this!
        return os.path.join(
            self.checkpoints_location,
            self.BEST_VAL_FOLDERNAME
        )

    @property
    def max_epoch(self):
        # heads up, this is slow, don't use it in an inner loop!
        return max(
            int(i[(len(self.BEST_VAL_FOLDERNAME) + 1):-len('.meta')])
            for i in os.listdir(self.checkpoints_location)
            if i.startswith(self.BEST_VAL_FOLDERNAME) and i.endswith('.meta')
        )

    @property
    def get_best_val_model_path(self):
        # heads up, this is slow, don't use it in an inner loop!
        return '{}-{}'.format(
            self.path_for_best_val_model,
            self.max_epoch
        )


class StatsKeeper():
    # Watch out! This is used in restoring models
    #

    def __init__(self, path, run_id, network_config):
        self.run_id = run_id
        self.path = path

        with open(path, 'w+') as f:
            f.write(network_config.serialized)
            f.write('\n')

    def restore_from_location(self, path):
        '''Given a path to an old json stats folder, transfers over its data to this StatsKeeper's
        stats folder.'''

        with open(path) as old_f:

            if path.endswith('stats.json'):
                # no backwards compatibility stuff yet! so we just need to copy the file over
                with open(self.path, 'a') as new_f:
                    for line in old_f:
                        new_f.write(line)
            else:
                raise Exception("File {} isn't a supported stats file!".format(path))


    def add_stat(self, epoch, train_stats, val_stats):
        data = {
            'epoch': epoch,
            'run_id': 'r' + self.run_id,
            'timestamp': int(time.time()),
            'train_c_loss': train_stats.loss,
            'train_c_accuracy': train_stats.accuracy,
            'val_c_loss': val_stats.loss,
            'val_c_accuracy': val_stats.accuracy,
        }

        with open(self.path, 'a') as f:
            json.dump(data, f)
            f.write('\n')
