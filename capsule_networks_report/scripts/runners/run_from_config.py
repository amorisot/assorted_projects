import argparse
from collections import namedtuple
import os.path
import time

import numpy as np
import tensorflow as tf
import tqdm

from data_loader_wrapper import datasets_from_provider_name
from tf_graph_helpers import parameters_for_config_id
from tf_graph_helpers import count_parameters
from loss_functions import build_train_ops
from loss_functions import build_losses_ops
from network_config import parse_network_config
from storage import FolderPicker
from storage import StatsKeeper
from storage import config_id_from_location
from important_metrics import EarlyStoppingActionEnum
from important_metrics import EarlyStoppingHelper
from important_metrics import accuracy_ops
from important_metrics import should_count_accuracy


DataSetStats = namedtuple('DataSetStats', ['loss', 'accuracy'])
LazyBatchProcessor = namedtuple('LazyBatchProcessor', ['data', 'mini_batch_func', 'data_name'])
LazyBatchProcessorPair = namedtuple('LazyBatchProcessorPair', [
    'train',
    'valid',
    'layer_features',
    'input_placeholder_for_gpus',
    'targets_placeholder_for_gpus',
    'is_training_placeholder',
    'losses_ops',
])


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Run the requested network.')
    parser.add_argument('--network-config', dest='network_config',
                    help='Configuration that defines the network')
    parser.add_argument('--output-location', dest='output_location',
                    help='Where files should be saved')
    parser.add_argument('--debug', dest='is_debug', action='store_true',
                    help='Don\'t run, just print actions and shapes')
    parser.add_argument('--require-gpu', dest='allow_soft_placement', action='store_false',
                    help='Requires things that should run on the GPU to run on the GPU.')
    parser.add_argument('--graph-only', dest='graph_only', action='store_true',
                    help='Like debug, but also save the graph')

    # These change the run
    parser.add_argument('--restore-from-location', dest='restore_from_location',
                    help='Pick up where the pointed-to-run left off.')
    parser.add_argument('--restore-from-location-config-folder', dest='restore_from_location_config_folder',
                    help='Config folder containing this config.', default='config')

    parser.add_argument('--num-gpus', dest='num_gpus', default=0, type=int)
    parser.add_argument('--seed', dest='seed', default=25012018, type=int)

    args = parser.parse_args(args)

    # extra arg parse logic
    if (
        (not args.is_debug and not args.output_location)
        and (not args.restore_from_location and not args.output_location)
    ):
        raise Exception('Need --output-location if not in debug mode or restore-from-location')

    if args.is_debug and args.graph_only:
        raise Exception('Only one of --debug and --graph-only allowed')

    return args


def run_epoch(sess, epoch, network_config, lazy_batch_processor, run_id):

    # TODO: stop reporting 0 accuracy
    total_c_loss = 0.
    total_accuracy = 0.

    batch_count = lazy_batch_processor.data.num_batches

    with tqdm.tqdm(total=batch_count) as pbar_train:
        for batch_idx, (input_batch_for_gpus, target_batches_for_gpus) in enumerate(lazy_batch_processor.data):

            run_output = lazy_batch_processor.mini_batch_func(sess, input_batch_for_gpus, target_batches_for_gpus)
            # Here we execute the c_error_opt_op which trains the network and also the ops that compute the
            # loss and accuracy, we save those in _, c_loss_value and acc respectively.
            total_c_loss += run_output.loss  # add loss of current iter to sum

            if run_output.maybe_accuracy is not None:
                total_accuracy += run_output.maybe_accuracy # add acc of current iter to sum

            # show iter statistics using running averages of previous iter within this epoch
            iter_id = epoch * batch_count + batch_idx
            iter_out = "r{run_id} | iter_num: {iter_id}, {data_name}_loss: {loss}, {data_name}_accuracy: {acc}".format(
                run_id=run_id,
                iter_id=iter_id,
                loss=total_c_loss / (batch_idx + 1),
                acc=total_accuracy / (batch_idx + 1),
                data_name=lazy_batch_processor.data_name,
            )
            pbar_train.set_description(iter_out)
            pbar_train.update(1)

    total_c_loss /= batch_count  # compute mean of loss
    total_accuracy /= batch_count # compute mean of accuracy

    return DataSetStats(total_c_loss, total_accuracy)


def verify_restore_location(network_config, restore_from_location):
    '''Checks that the location we're restoring from is a run
    from this experiment
    '''

    # if this ends with a '/', delete it
    if restore_from_location[-1] == '/':
        restore_from_location = restore_from_location[:-1]

    parts = restore_from_location.split('/')

    run_id = str(int(parts[-1]))  # This should point at a run, which is an string of an int.

    if parts[-2] != network_config.config_id:
        raise ValueError(
            'restore location {} is a different experiment than config {}'.format(
                    restore_from_location,
                    network_config.config_id
                )
            )

    output_location = '/'.join(parts[:-2])

    return output_location, run_id


def get_restorer_folder_picker(network_config, restore_from_location):
    output_location, run_id = verify_restore_location(network_config, restore_from_location)
    return FolderPicker(output_location, network_config, run_id, mkdirs=False, restorer=True)


RunOutput = namedtuple('RunOutput', [
    'loss',
    'maybe_accuracy', # Optional
])


def build_network(network_config):
    # We'll need the rng to get datasets
    rng = np.random.RandomState(seed=network_config.seed)

    # data_splits also grabs input and output placeholders
    data_splits, input_placeholder_for_gpus, targets_placeholder_for_gpus = datasets_from_provider_name(rng, network_config)

    is_training_placeholder = tf.placeholder(tf.bool, name='training-flag')

    # produce losses and get layer features to save for visual inspection
    with tf.name_scope("build_losses_ops"):
        losses_op_by_gpu, layer_features = build_losses_ops(
            network_config,
            input_placeholder_for_gpus,
            targets_placeholder_for_gpus,
            is_training_placeholder,
            data_splits,
            reuse=False,
        )

    # TODO: the target_placeholder slice is hacky/hidden
    acc_ops = tf.no_op()
    if should_count_accuracy(network_config):
        with tf.name_scope("build_accuracy_ops"):
            target_placeholders = targets_placeholder_for_gpus
            if len(targets_placeholder_for_gpus.shape) > 2:
                target_placeholders = targets_placeholder_for_gpus[:, :, 0]

            acc_ops = accuracy_ops(
                network_config,
                layer_features,
                target_placeholders, # the first column should always be the class
            )

    # grab and print variables
    variables = parameters_for_config_id(network_config)
    for variable in variables:
        print(variable)
    print('total of {} variables'.format(count_parameters(variables)))

    print('optimizing with adam using {}'.format(network_config.optimizer_kwargs))
    with tf.name_scope("build_train_ops"):
        c_error_opt_op = build_train_ops(
            network_config,
            variables,
            losses_op_by_gpu,
            **network_config.optimizer_kwargs._asdict(),
        )

        total_losses_over_gpu = tf.reduce_mean(losses_op_by_gpu, axis=0)
        # save summaries for the losses
        # and the layer
        tf.summary.scalar('crossentropy_losses', total_losses_over_gpu)

        summary_op = tf.summary.merge_all()

    # The function returned will be called on every minibatch
    def get_trainer_runner_func(is_training):

        if is_training:
            # only evaluate c_error_opt_op if we want to train.
            variables_to_eval = [c_error_opt_op, total_losses_over_gpu, acc_ops]
            data = data_splits.train_data
            data_name = 'train'

        else:
            variables_to_eval = [total_losses_over_gpu, acc_ops]
            data = data_splits.val_data
            data_name = 'val'

        def func(sess, input_batch, target_batch):
            result = sess.run(
                variables_to_eval,
                feed_dict={
                    input_placeholder_for_gpus: input_batch,
                    targets_placeholder_for_gpus: target_batch,
                    is_training_placeholder: is_training
                })

            # By indexing from the end, skip `c_error_opt_op`
            return RunOutput(
                loss=result[-2],
                maybe_accuracy=result[-1],
            )

        return LazyBatchProcessor(
            data=data,
            mini_batch_func=func,
            data_name=data_name,
        )

    return LazyBatchProcessorPair(
        layer_features=layer_features,
        input_placeholder_for_gpus=input_placeholder_for_gpus,
        targets_placeholder_for_gpus=targets_placeholder_for_gpus,
        is_training_placeholder=is_training_placeholder,
        losses_ops=losses_op_by_gpu,
        train=get_trainer_runner_func(is_training=True),
        valid=get_trainer_runner_func(is_training=False),
    )


def run_single_batch(lazy_batch_processor):
    input_batch_for_gpus, target_batches_for_gpus = next(lazy_batch_processor.data)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        c_loss_value, acc = lazy_batch_processor.mini_batch_func(sess, input_batch_for_gpus, target_batches_for_gpus)

    print('finished!')


def run_from_config(network_config, args):

    print(network_config.serialized)

    lazy_batch_processor_pair = build_network(network_config)

    # early return if we're just debugging
    if args.is_debug:
        return

    # Otherwise, start setting up the run

    # Now set up the run_id and folder-picker!
    run_id = str(int(time.time()))

    folder_picker = FolderPicker(args.output_location, network_config, run_id, mkdirs=True)

    tf_file_writer = tf.summary.FileWriter(
        folder_picker.path_for_tf_logs_dir,
        tf.get_default_graph()
    )
    print('writing tensorflow logs to {}'.format(folder_picker.path_for_tf_logs_dir))
    print('view with \n    tensorboard --logdir {}'.format(folder_picker.path_for_tf_logs_dir))
    if args.graph_only:
        run_single_batch(lazy_batch_processor_pair.train)
        return

    print('stats will be saved in', folder_picker.path_for_epoch_stats)

    early_stopping_helper = EarlyStoppingHelper(network_config)

    stat_keeper = StatsKeeper(folder_picker.path_for_epoch_stats, run_id, network_config)

    init = tf.global_variables_initializer()
    with tf.Session(
        config=tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
        )
    ) as sess:
        sess.run(init)
        train_saver = tf.train.Saver()
        val_saver = tf.train.Saver()  # The provided code has two savers! I'm not sure why yet.

        # If a folder is specified, load up the model and copy over the stats file.
        start_epoch = 0
        if args.restore_from_location:
            restorer_fp = get_restorer_folder_picker(network_config, args.restore_from_location)
            start_epoch = restorer_fp.max_epoch + 1
            checkpoint_location = restorer_fp.get_best_val_model_path
            stat_location = restorer_fp.path_for_epoch_stats

            print('restoring epoch {} from {}'.format(
                start_epoch,
                checkpoint_location
            ))
            stat_keeper.restore_from_location(stat_location)
            train_saver.restore(sess, checkpoint_location)

        # For each epoch, run the training and validation set.
        with tqdm.tqdm(total=network_config.epochs - start_epoch) as epoch_pbar:
            for e in range(start_epoch, network_config.epochs):

                train_epoch_stats = run_epoch(
                    sess,
                    e,
                    network_config,
                    lazy_batch_processor_pair.train,
                    run_id,
                )

                val_epoch_stats = run_epoch(
                    sess,
                    e,
                    network_config,
                    lazy_batch_processor_pair.valid,
                    run_id,
                )

                # and store the epoch's stats
                stat_keeper.add_stat(e, train_stats=train_epoch_stats, val_stats=val_epoch_stats)

                # Store the best model for early stopping
                early_stopping_action = early_stopping_helper.action_given_epoch_stats(e, val_epoch_stats)
                if early_stopping_action == EarlyStoppingActionEnum.SAVE:
                    save_path = val_saver.save(sess, folder_picker.path_for_best_val_model, global_step=e)
                    print("Saved best validation score model at", save_path)
                    print()
                elif early_stopping_action == EarlyStoppingActionEnum.STOP:
                    print('No improvement in a while. Quitting!')
                    break
                elif early_stopping_action == EarlyStoppingActionEnum.CONTINUE:
                    continue
                else:
                    raise ValueError('unrecognized stopping action `{}`'.format(early_stopping_action))

                epoch_pbar.update(1)

        save_path = train_saver.save(sess, folder_picker.path_for_trained_model)
        print("Saved training model at", save_path)

    tf_file_writer.close()


def _network_config_location_from_args(args):
    if args.network_config is not None:
        return args.network_config

    elif args.restore_from_location is not None:
        # extract config location from name
        config_id = config_id_from_location(args.restore_from_location)

        return os.path.join(args.restore_from_location_config_folder, config_id) + '.yaml'

    else:
        raise Exception('no network config or restore from location')


def main():
    args = parse_args()

    network_config_location = _network_config_location_from_args(args)
    print('Loading from {}'.format(network_config_location))

    network_config = parse_network_config(network_config_location, args)

    run_from_config(
        network_config,
        args,
    )


if __name__ == '__main__':
    main()
