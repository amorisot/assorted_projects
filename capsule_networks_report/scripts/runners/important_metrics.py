'''Early stopping and metrics important for early stopping (like accuracy).
For loss functions, see loss_functions.py'''
import tensorflow as tf

from loss_functions import device_ids_from_network_config


def should_count_accuracy(network_config):
    return type(network_config.important_metric).__name__ == 'AccuracyMetric'


def accuracy_ops(
    network_config,
    layer_features,
    targets_placeholder_for_gpus,
):

    device_ids = device_ids_from_network_config(network_config)

    with tf.name_scope("accuracy_ops"):
        accuracies = []
        for i, device_id in enumerate(device_ids):
            with tf.device(device_id):
                with tf.name_scope("accuracy_{}".format(i)):
                    target_placeholder = targets_placeholder_for_gpus[i]

                    preds = layer_features[network_config.important_metric.prediction_layer_idx].ongoing_result

                    if not preds.shape == target_placeholder.shape:
                        raise Exception('prediction and target shapes are different')

                    correct_prediction = tf.equal(preds, tf.cast(target_placeholder, tf.int64))
                    accuracies.append(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))

        accuracy = tf.reduce_mean(tf.cast(accuracies, tf.float32))

        # add loss and accuracy to collections
        tf.add_to_collection('accuracy', accuracy)
        tf.summary.scalar('accuracy', accuracy)

        total_accuracy = tf.add_n(tf.get_collection('accuracy'), name='total_accuracy')

    return total_accuracy


class EarlyStoppingActionEnum():
    SAVE = 'save'
    CONTINUE = 'continue'
    STOP = 'stop'


class EarlyStoppingHelper():
    '''This tells you if you should save, continue, or stop.
    every step in the epoch, update it with the validation set accuracy.
    '''

    def __init__(self, network_config):
        self.best_val_metric = None
        self.best_epoch = 0
        self.early_stop_after_n_epochs = network_config.early_stop_after_n_epochs

        # start the time since improvement at the max
        self.time_since_improvement = self.early_stop_after_n_epochs

        # get function for extracting early stopping value
        self._metric_from_run_output = self._pick_important_metric(network_config)

        self._metric_improved = self._pick_if_important_metric_increase(network_config)

    def action_given_epoch_stats(self, epoch, epoch_stats):
        '''Checks whether accuracy is the best so far. Assumes
        you'll save this model.'''

        metric = self._metric_from_run_output(epoch_stats)

        # first run, so save the number and keep going
        if self.best_val_metric is None:
            self.best_val_metric = metric
            return EarlyStoppingActionEnum.CONTINUE

        elif self._metric_improved(self.best_val_metric, metric):
            self.best_val_metric = metric
            self.best_epoch = epoch
            self.time_since_improvement = self.early_stop_after_n_epochs
            return EarlyStoppingActionEnum.SAVE

        elif self.time_since_improvement is not None:
            if self.time_since_improvement <= 0:
                # This was the last step, so we're done
                return EarlyStoppingActionEnum.STOP
            else:
                # otherwise, decrease time since improvement and keep going
                self.time_since_improvement -= 1
                return EarlyStoppingActionEnum.CONTINUE

        else:
            return EarlyStoppingActionEnum.CONTINUE

    def _pick_important_metric(self, network_config):
        if should_count_accuracy(network_config):
            return lambda x: x.accuracy
        else:
            return lambda x: x.loss

    def _pick_if_important_metric_increase(self, network_config):
        # accuracy should increase, so it's improved if new > old
        if should_count_accuracy(network_config):
            return lambda old, new: new > old
        else:
            return lambda old, new: old > new
