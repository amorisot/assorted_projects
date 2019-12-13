'''To run these, run the following

pip install pytest
pytest (this file name)

'''
import os

import tensorflow as tf
import pytest
import numpy as np

from data_loader_wrapper import DataSplits
from run_from_config import EarlyStoppingHelper
from run_from_config import SAVE_EARLY_STOPPING_ACTION
from run_from_config import CONTINUE_EARLY_STOPPING_ACTION
from run_from_config import STOP_EARLY_STOPPING_ACTION
from run_from_config import run_from_config
from run_from_config import verify_restore_location
from run_from_config import run_epoch
import layer_models
from network_config import CrossEntropyLossInfo
from network_config import loss_info_from_dict
from network_config import NetworkConfig
from network_config import optimizer_info_from_dict
from network_graph import parameters_for_config_id
from network_graph import count_parameters
from network_graph import accuracy_ops
from storage import FolderPicker


@pytest.fixture()
def mock_config_id():
    return 'test_config_id'

@pytest.fixture()
def mock_network_config(mock_config_id):
    layers = layer_models.layers_from_list([
        {'type': 'softmax_pred', 'num_classes': 10, 'id': 'prediction'},
        {'type': 'argmax'},
    ])
    return NetworkConfig(
        config_id=mock_config_id,
        data_provider_name='cifar10',
        batch_size=1,
        seed=1,
        loss=loss_info_from_dict(
            {'type': 'softmax_crossentropy', 'softmax_id': 'prediction'},
            layers,
        ),
        epochs=1,
        layers=layers,
        early_stop_after_n_epochs=1,
        prediction_layer_idx=-1,  # We don't use this, so put garbage number
        optimizer_kwargs={},
    )

# test early stopping helper

class TestEarlyStoppingHelper():
    EARLY_STOP_AFTER_N = 3

    def test_smoke(self):
        es = EarlyStoppingHelper(self.EARLY_STOP_AFTER_N)

        # first add a case where we should save
        assert es.action_given_accuracy(0, 0.5) == SAVE_EARLY_STOPPING_ACTION

        # then add something with a worse accuracy
        assert es.action_given_accuracy(1, 0) == CONTINUE_EARLY_STOPPING_ACTION

        # then add another thing with accuracy, but not as good as the first time
        assert es.action_given_accuracy(2, 0.25) == CONTINUE_EARLY_STOPPING_ACTION

        # then add something with great accuracy
        assert es.action_given_accuracy(3, 1) == SAVE_EARLY_STOPPING_ACTION

        # now do EARLY_STOP_AFTER_N rounds of worse accuracy
        for i in range(self.EARLY_STOP_AFTER_N):
            assert es.action_given_accuracy(4 + i, 0) == CONTINUE_EARLY_STOPPING_ACTION

        # then the early stopping should kick in
        assert es.action_given_accuracy(4 + self.EARLY_STOP_AFTER_N, 0) == STOP_EARLY_STOPPING_ACTION

        # and if accidentally keep going, it should continue giving the same result
        assert es.action_given_accuracy(5 + self.EARLY_STOP_AFTER_N, 0) == STOP_EARLY_STOPPING_ACTION

        # and the best epoch should still be 3
        assert es.best_epoch == 3

    def test_disabled(self):
        es = EarlyStoppingHelper()

        # the only difference in this case is that it never says to stop

        # first add a case where we should save
        assert es.action_given_accuracy(0, 0.5) == SAVE_EARLY_STOPPING_ACTION

        # then add something with a worse accuracy
        assert es.action_given_accuracy(1, 0) == CONTINUE_EARLY_STOPPING_ACTION

        # then add another thing with accuracy, but not as good as the first time
        assert es.action_given_accuracy(2, 0.25) == CONTINUE_EARLY_STOPPING_ACTION

        # then add something with great accuracy
        assert es.action_given_accuracy(3, 1) == SAVE_EARLY_STOPPING_ACTION

        # now do EARLY_STOP_AFTER_N rounds of worse accuracy
        for i in range(100):
            assert es.action_given_accuracy(4 + i, 0) == CONTINUE_EARLY_STOPPING_ACTION



class TestCapsnet():
    CAPSNET_CONFIG = 'config/sample_capsnet.yaml'

    def test_var_count(self, tmpdir):
        p = tmpdir.mkdir("capsnettest")

        network_config = NetworkConfig.parse_config(self.CAPSNET_CONFIG)

        run_from_config(
            str(p),
            network_config,
            True,
            None,
        )

        variables = parameters_for_config_id(network_config)

        # mnist capsnet set up like aguron's notebook should have this many params
        assert count_parameters(variables) == 8215568


class TestRestore():
    def test_bad_last_folder(self, mock_config_id, mock_network_config):
        with pytest.raises(ValueError):
            verify_restore_location(mock_network_config, mock_config_id + '/somethingelse')

    def test_bad_last_folder(self, mock_config_id, mock_network_config):
        with pytest.raises(ValueError):
            verify_restore_location(mock_network_config, 'something/123')

    def test_good(self, mock_config_id, mock_network_config):
        assert verify_restore_location(mock_network_config, 'b/a/' + mock_config_id + '/123') == ('b/a', '123')

    def test_good_trailing_slash(self, mock_config_id, mock_network_config):
        assert verify_restore_location(mock_network_config, 'b/a/' + mock_config_id + '/123/') == ('b/a', '123')

# Make a test for each config
@pytest.mark.parametrize("config_path", [
    os.path.join('config/', filename)
    for filename in os.listdir('config/')
])
def test_all_configs(tmpdir, config_path):
    # This test just crashes if one of the configs can't be
    # loaded or built

    p = tmpdir.mkdir("smoke_test")
    print(config_path)
    tf.reset_default_graph()

    network_config = NetworkConfig.parse_config(config_path)

    run_from_config(
        str(p),
        network_config,
        is_debug=True,
    )

    variables = parameters_for_config_id(network_config)

    print(config_path, count_parameters(variables))


class TestAccuracy():

    def test_compute_accuracy_with_diff_shapes(self, mock_network_config):
        targets_placeholder = tf.placeholder(
            np.int64,
            [mock_network_config.batch_size],
            'data-targets'
        )

        fake_inputs = np.ones((1, 32, 32)).astype(np.int64)
        fake_targets = np.ones(1)

        with tf.Session() as sess:
            with pytest.raises(Exception):
                acc = accuracy_ops(
                    mock_network_config,
                    targets_placeholder,
                    [layer_models.LayerResult(None, fake_inputs)],
                )

                acc.eval(feed_dict={targets_placeholder: fake_targets})

    def test_compute_accuracy(self, mock_network_config):
        batch_size = 2

        targets_placeholder = tf.placeholder(
            np.int64,
            [batch_size],
            'data-targets'
        )

        fake_inputs = np.ones((batch_size)).astype(np.int64)
        fake_targets = np.ones(batch_size)

        fake_targets_bad = np.zeros(batch_size)

        fake_targets_both = np.hstack((
            np.zeros(1),
            np.ones(1),
        ))

        with tf.Session() as sess:
            acc = accuracy_ops(
                mock_network_config,
                targets_placeholder,
                [layer_models.LayerResult(None, fake_inputs)],
            )

            assert np.isclose(acc.eval(feed_dict={targets_placeholder: fake_targets}), 1)
            assert np.isclose(acc.eval(feed_dict={targets_placeholder: fake_targets_bad}), 0)
            assert np.isclose(acc.eval(feed_dict={targets_placeholder: fake_targets_both}), 0.5)

class MockDataProivder():


    def __init__(self, data):
        self.data = data
        self.i = 0
        self.num_batches = len(data)

    def __next__(self):
        if self.i >= len(self.data):
            raise StopIteration()
        d = self.data[self.i]
        self.i += 1
        return d

    def __iter__(self):
        return self


class TestRunEpoch():
    NUM_BATCHES = 10

    def test_run_epoch_0s(self, mock_network_config):

        fake_data = np.zeros((self.NUM_BATCHES, 2))

        data_splits = DataSplits(
            train_data=MockDataProivder(fake_data),
            val_data=[],
            test_data=[],
        )

        def runner_func(input_batch, target_batch):
            return (0, 0)

        stats = run_epoch(
            1,
            mock_network_config,
            data_splits,
            'train_data',
            runner_func
        )

        assert stats.accuracy == 0
        assert stats.loss == 0


    def test_run_epoch_mix(self, mock_network_config):

        fake_data = np.zeros((self.NUM_BATCHES, 2))

        data_splits = DataSplits(
            train_data=MockDataProivder(fake_data),
            val_data=[],
            test_data=[],
        )

        FAKE_ACCURACY = 10
        FAKE_LOSS = -55

        def runner_func(input_batch, target_batch):
            return (FAKE_LOSS, FAKE_ACCURACY)

        stats = run_epoch(
            1,
            mock_network_config,
            data_splits,
            'train_data',
            runner_func
        )

        # The average of a bunch of things that are the same is that thing
        assert stats.accuracy == FAKE_ACCURACY
        assert stats.loss == FAKE_LOSS


class TestFolderPicker():
    def test_stats_folder_race_condition(self, tmpdir, mock_network_config):
        p = tmpdir.mkdir("capsnettest")

        # First time should be fine
        FolderPicker(str(p), mock_network_config, '123')

        # second time should raise
        with pytest.raises(Exception):
            FolderPicker(str(p), mock_network_config, '123')


def test_optimizer_info_from_dict():
    assert optimizer_info_from_dict({})._asdict() == {
        'learning_rate': 1e-3,
        'beta1': 0.9,
    }

    assert optimizer_info_from_dict(
        {'learning_rate': 5})._asdict() == {
        'learning_rate': 5,
        'beta1': 0.9,
    }

    assert optimizer_info_from_dict(
        {'beta1': 5})._asdict() == {
        'learning_rate': 1e-3,
        'beta1': 5,
    }

    with pytest.raises(Exception):
        optimizer_info_from_dict({'fake key': 1})
