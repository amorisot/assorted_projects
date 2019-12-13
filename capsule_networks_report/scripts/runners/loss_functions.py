'''

Note: most of this is copy-pasta'd from the provided code!
'''
from collections import namedtuple

import tensorflow as tf
import numpy as np

from layer_models import get_result_of_layer_id
from layer_models import safe_norm
from layer_models import build_features


# Config info needed for loss different loss functions
LOSS_INFO_REQUIRED_FIELDS = ['func']


def device_ids_from_network_config(network_config):
    # makes an ordered list of devices to send ops off to
    if network_config.use_cpu:
        device_ids = ['/cpu:0']
    else:
        device_ids = ['/gpu:{}'.format(i) for i in range(network_config.num_devices)]

    return device_ids


# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def build_train_ops(network_config, variables, losses_op_by_gpu, learning_rate, beta1):
    device_ids = device_ids_from_network_config(network_config)

    opts_ops = []

    # tell all the boxes to minimize their corresponding device_id
    for device_id, losses_op in zip(device_ids, losses_op_by_gpu):
        c_opt = tf.train.AdamOptimizer(beta1=beta1, learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        with tf.control_dependencies(update_ops):
            opts_ops.append(
                c_opt.compute_gradients(losses_op, var_list=variables, colocate_gradients_with_ops=True))

    # then bring everything together
    grads = average_gradients(opts_ops)
    # and apply the gradients
    train_ops = c_opt.apply_gradients(grads)

    return train_ops

def build_losses_ops(
    network_config,
    input_placeholder_for_gpus,
    targets_placeholder_for_gpus,
    is_training_placeholder,
    data_splits,
    reuse=False,
):

    with tf.name_scope("losses_ops"):
        # mult-task learning has an extra dimension
        class_targets_placeholder = targets_placeholder_for_gpus
        if len(targets_placeholder_for_gpus.get_shape()) > 2:
            class_targets_placeholder = targets_placeholder_for_gpus[:, :, 0]

        # build the features and losses on each device
        should_reuse = reuse
        device_ids = device_ids_from_network_config(network_config)
        losses_op_by_gpu = []
        for i, device_id in enumerate(device_ids):
            with tf.device(device_id):

                with tf.name_scope("features_{}".format(i)):
                    layer_features, _ = build_features(
                        network_config,
                        input_placeholder_for_gpus[i],
                        class_targets_placeholder[i], # build the features using the class targets
                        is_training_placeholder,
                        data_splits,
                        reuse=should_reuse,
                    )

                with tf.name_scope("losses_{}_{}".format(type(network_config.loss).__name__, i)):
                    losses_op_by_gpu.append(
                        network_config.loss.func(
                            network_config,
                            input_placeholder_for_gpus[i],  # only use input/targets for this device
                            targets_placeholder_for_gpus[i],  # use all targets for the loss function
                            is_training_placeholder,
                            data_splits,
                            layer_features,
                            reuse=should_reuse,
                        )
                    )
                # reuse variables for the next device
                should_reuse = True

    return losses_op_by_gpu, layer_features

def assert_scalar(tensor):
    assert tensor.get_shape().as_list() == [], tensor.get_shape().as_list()

## Begin loss functions

def compute_reconstruction_loss(input_placeholder, decoder_output):

    batch_size, input_x, input_y, num_channels = input_placeholder.get_shape().as_list()

    n_output = input_x * input_y

    X_flat = tf.reshape(input_placeholder, [-1, n_output * num_channels], name="X_flat")
    squared_difference = tf.square(X_flat - decoder_output,
                                   name="squared_difference")
    reconstruction_loss = tf.reduce_mean(squared_difference,
                                        name="reconstruction_loss")

    return reconstruction_loss


def compute_azimith_loss(azimith_targets_placeholder, azimith_pred, azimith_func_name):
    # Heads up! Some early jobs tried to predict the angle (as in on the order of 360). I changed this
    # because it seemed like it was having a hard time predicting big numbers. Instead the number
    # that comes out of the last layer should be fraction of angle out of 360.
    degree_difference = tf.cast(azimith_targets_placeholder, tf.float32) - (azimith_pred[:, 0] * 360)

    # tf.sin needs radians
    radian_difference = degree_difference / 180 * np.pi

    if azimith_func_name == 'sin_squared':
        # sin^2(x/2)
        azimith_func_result = tf.pow(tf.sin(radian_difference / 2), 2)

    elif azimith_func_name == 'adrien':
        #  sin^2(x) - 0.5 sin^2(x/2 + pi)
        azimith_func_result = tf.add(
            tf.pow(tf.sin(radian_difference), 2),
            -0.5 * tf.pow(tf.sin(radian_difference / 2 + np.pi), 2),
        )
    else:
        raise Exception('not a valid azimith func: {}'.format(azimith_func_name))

    average_azimith_loss = tf.reduce_mean(azimith_func_result)
    final_azimith_loss = average_azimith_loss

    return final_azimith_loss


MTLCapsNetLossInfo = namedtuple('MTLCapsNetLossInfo', LOSS_INFO_REQUIRED_FIELDS + [
    'caps_routing_id',  # last capsule net routing output
    'decoder_output_id',  # capsule from decoder
    'azimith_output_id',  # azimith output layer id
    'azimith_func',  # name of function to compute loss on angles
    'alpha',  # how much to weight the reconstruction loss
    'azimith_weight',  # how much to weight the azimith part of the loss function
])

MTL_CAPSNET_LOSS_INFO_DEFAULTS = {}


def build_MTL_caps_azimith_margin_losses_ops(
    network_config,
    input_placeholder,
    targets_placeholder,
    is_training,
    data_splits,
    layer_features,
    reuse=True,
):

    class_targets_placeholder = targets_placeholder[:, 0]
    azimith_targets_placeholder = targets_placeholder[:, 1]

    # start by computing the loss on the regular capsnet loss on the first
    # field of the targets
    base_loss = build_caps_margin_losses_ops(
        network_config,
        input_placeholder,
        class_targets_placeholder,
        is_training,
        data_splits,
        layer_features,
        reuse=reuse,
    )

    # now compute the loss function on the azimith task
    with tf.name_scope("azimith_loss"):
        azimith_pred = get_result_of_layer_id(
            network_config.loss.azimith_output_id,
            layer_features,
        )
        final_azimith_loss = compute_azimith_loss(azimith_targets_placeholder, azimith_pred, network_config.loss.azimith_func)

    with tf.name_scope("combine_losses"):
        combined_margin_azimith = tf.add(base_loss, network_config.loss.azimith_weight * final_azimith_loss, name='combined_margin_azimith')

        assert_scalar(combined_margin_azimith)

    return combined_margin_azimith


CapsNetLossInfo = namedtuple('CapsNetLossInfo', LOSS_INFO_REQUIRED_FIELDS + [
    'caps_routing_id',  # last capsule net routing output
    'decoder_output_id',  # capsule from decoder
    'alpha',  # how much to weight the reconstruction loss
])

CAPSNET_LOSS_INFO_DEFAULTS = {'alpha': 0.0005}

# As usual, caps net stuff is copied from
# https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb
def build_caps_margin_losses_ops(
    network_config,
    input_placeholder,
    targets_placeholder,
    is_training,
    data_splits,
    layer_features,
    reuse=True,
):

    # set some constants
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    batch_size, input_x, input_y, num_channels = input_placeholder.get_shape().as_list()

    # targets should only contain a column of classes
    assert targets_placeholder.get_shape().as_list() == [batch_size,]

    n_classes = data_splits.train_data.num_classes

    with tf.name_scope("caps_margin_losses"):
        T = tf.one_hot(tf.cast(targets_placeholder, tf.int64), depth=n_classes, name="T")

        caps2_output = get_result_of_layer_id(
            network_config.loss.caps_routing_id,
            layer_features,
        )

        caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                                      name="caps2_output_norm")

        present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                                      name="present_error_raw")
        present_error = tf.reshape(present_error_raw, shape=(-1, n_classes),
                                   name="present_error")

        absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                                     name="absent_error_raw")
        absent_error = tf.reshape(absent_error_raw, shape=(-1, n_classes),
                                  name="absent_error")

        L = tf.add(
            T * present_error,
            lambda_ * (1.0 - T) * absent_error,
            name="L")

        margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

    with tf.name_scope('reconstruction_loss'):

        decoder_output = get_result_of_layer_id(
            network_config.loss.decoder_output_id,
            layer_features,
        )

        reconstruction_loss = compute_reconstruction_loss(input_placeholder, decoder_output)

    with tf.name_scope('final_loss'):
        final_losses = tf.add(margin_loss, network_config.loss.alpha * reconstruction_loss, name="loss")

        tf.add_to_collection('final_losses', final_losses)

        total_loss = tf.add_n(tf.get_collection('final_losses'), name='total_loss')

        assert_scalar(total_loss)

    return total_loss


MTLReconstructionLossInfo = namedtuple('MTLReconstructionLossInfo', LOSS_INFO_REQUIRED_FIELDS + [
    'decoder_output_id',
    'azimith_output_id',
    'softmax_output_id',
    'azimith_func',
    'reconstruction_weight',
    'azimith_weight',
])

MTL_RECONSTRUCTION_LOSS_INFO_DEFAULTS = {}


def build_mtl_reconstruction_losses_ops(
    network_config,
    input_placeholder,
    targets_placeholder,
    is_training,
    data_splits,
    layer_features,
    reuse=True,
):

    batch_size, input_x, input_y, num_channels = input_placeholder.get_shape().as_list()

    class_targets_placeholder = targets_placeholder[:, 0]
    azimith_targets_placeholder = targets_placeholder[:, 1]

    n_classes = data_splits.train_data.num_classes
    n_output = input_x * input_y

    with tf.name_scope('crossentropy_loss'):
        preds = get_result_of_layer_id(
            network_config.loss.softmax_output_id,
            layer_features,
        )
        crossentropy_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=class_targets_placeholder,
                logits=preds,
            )
        )

    with tf.name_scope('reconstruction_loss'):
        decoder_output = get_result_of_layer_id(
            network_config.loss.decoder_output_id,
            layer_features,
        )
        reconstruction_loss = compute_reconstruction_loss(input_placeholder, decoder_output)

    # now compute the loss function on the azimith task
    with tf.name_scope("azimith_loss"):
        azimith_pred = get_result_of_layer_id(
            network_config.loss.azimith_output_id,
            layer_features,
        )
        final_azimith_loss = compute_azimith_loss(azimith_targets_placeholder, azimith_pred, network_config.loss.azimith_func)

    with tf.name_scope('final_loss'):
        final_losses = tf.add(
            crossentropy_loss,
            tf.add(
                network_config.loss.reconstruction_weight * reconstruction_loss,
                network_config.loss.azimith_weight * final_azimith_loss,
            )
        )

        tf.add_to_collection('final_losses', final_losses)

        total_loss = tf.add_n(tf.get_collection('final_losses'), name='total_loss')

        assert_scalar(total_loss)

    return total_loss


ReconstructionLossInfo = namedtuple('ReconstructionLossInfo', LOSS_INFO_REQUIRED_FIELDS + [
    'decoder_output_id',  # capsule from decoder
    'softmax_output_id',
    'reconstruction_weight',
])

RECONSTRUCTION_LOSS_INFO_DEFAULTS = {}


def build_reconstruction_losses_ops(
    network_config,
    input_placeholder,
    targets_placeholder,
    is_training,
    data_splits,
    layer_features,
    reuse=True,
):

    batch_size, input_x, input_y, num_channels = input_placeholder.get_shape().as_list()

    class_targets_placeholder = targets_placeholder

    n_classes = data_splits.train_data.num_classes
    n_output = input_x * input_y

    with tf.name_scope('crossentropy_loss'):
        preds = get_result_of_layer_id(
            network_config.loss.softmax_output_id,
            layer_features,
        )
        crossentropy_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=class_targets_placeholder,
                logits=preds,
            )
        )

    with tf.name_scope('reconstruction_loss'):
        decoder_output = get_result_of_layer_id(
            network_config.loss.decoder_output_id,
            layer_features,
        )
        reconstruction_loss = compute_reconstruction_loss(input_placeholder, decoder_output)

    with tf.name_scope('final_loss'):
        final_losses = tf.add(
            crossentropy_loss,
            network_config.loss.reconstruction_weight * reconstruction_loss,
        )

        tf.add_to_collection('final_losses', final_losses)

        total_loss = tf.add_n(tf.get_collection('final_losses'), name='total_loss')

        assert_scalar(total_loss)

    return total_loss


OnlyReconstructionLossInfo = namedtuple('OnlyReconstructionLossInfo', LOSS_INFO_REQUIRED_FIELDS + [
    'decoder_output_id',
])

ONLY_RECONSTRUCTION_LOSS_INFO_DEFAULTS = {}

def build_only_reconstruction_losses_ops(
    network_config,
    input_placeholder,
    targets_placeholder,
    is_training,
    data_splits,
    layer_features,
    reuse=True,
):

    with tf.name_scope('reconstruction_loss'):
        decoder_output = get_result_of_layer_id(
            network_config.loss.decoder_output_id,
            layer_features,
        )
        reconstruction_loss = compute_reconstruction_loss(input_placeholder, decoder_output)

    return reconstruction_loss


CrossEntropyLossInfo = namedtuple('CrossEntropyLoss', LOSS_INFO_REQUIRED_FIELDS + [
    'softmax_id'  # layer that outputs the softmax results
])

CROSS_ENTROPY_LOSS_INFO_DEFAULTS = {}

def build_softmax_crossentropy_losses_op(
    network_config,
    input_placeholder,
    targets_placeholder,
    is_training,
    data_splits,
    layer_features,
    reuse=True,
):
    """build models, calculates losses, saves summary statistcs and images.
    Returns:
        dict of losses.
    """
    with tf.name_scope("softmax_crossentropy_losses"):
        preds = layer_features[network_config.loss.softmax_id].ongoing_result

        # compute loss and accuracy
        crossentropy_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets_placeholder, logits=preds))

        # add loss and accuracy to collections
        tf.add_to_collection('crossentropy_losses', crossentropy_loss)

        total_classification_loss = tf.add_n(tf.get_collection('crossentropy_losses'), name='total_classification_loss')

    return total_classification_loss


LOSS_TYPE_TO_FUNC_AND_INFO_TUPLE = {
    'softmax_crossentropy': (build_softmax_crossentropy_losses_op, CrossEntropyLossInfo, CROSS_ENTROPY_LOSS_INFO_DEFAULTS),
    'caps_net': (build_caps_margin_losses_ops, CapsNetLossInfo, CAPSNET_LOSS_INFO_DEFAULTS),
    'mtl_caps_net': (build_MTL_caps_azimith_margin_losses_ops, MTLCapsNetLossInfo, MTL_CAPSNET_LOSS_INFO_DEFAULTS),
    'reconstruction': (build_reconstruction_losses_ops, ReconstructionLossInfo, RECONSTRUCTION_LOSS_INFO_DEFAULTS),
    'mtl_reconstruction': (build_mtl_reconstruction_losses_ops, MTLReconstructionLossInfo, MTL_RECONSTRUCTION_LOSS_INFO_DEFAULTS),
    'only_reconstruction': (build_only_reconstruction_losses_ops, OnlyReconstructionLossInfo, ONLY_RECONSTRUCTION_LOSS_INFO_DEFAULTS),
}
