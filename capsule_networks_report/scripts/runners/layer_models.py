'''This file converts the yaml (see network_config) into tensorflow actions.

The main entrypoint is the layer_from_dict. It looks at the "type" field, and uses
that to pick out which Layer class to build.

To add more activation functions, we should be able to just add another entry into
ACTIVATION_FROM_NAME.
'''
from collections import namedtuple

import tensorflow as tf
import numpy as np

from tensorflow.python.ops.nn_ops import leaky_relu
from tensorflow.python.ops.nn_ops import relu
from tensorflow.contrib.layers import batch_norm


OPTIONAL_DEFAULT_LAYER_RESULT_FIELDS = [
    # `assert_output_shape`: require the shape of the output from this layer,
    #   to be this shape (ignoring the batch_size)
    'assert_output_shape',
    # an id that can be used in things like the loss function or for skip connections
    'id',
    # usually data passes through each layer. With an input id, the input is
    # from the output of the provided layer
    'input_id',
]

REQUIRED_DEFAULT_LAYER_RESULT_FIELDS = []

DEFAULT_LAYER_RESULT_FIELDS = OPTIONAL_DEFAULT_LAYER_RESULT_FIELDS + REQUIRED_DEFAULT_LAYER_RESULT_FIELDS

ACTIVATION_FROM_NAME = {
    'leaky_relu': leaky_relu,
    'relu': relu,
    'sigmoid': tf.nn.sigmoid,
}


def activation_from_name(activation_name):
    # If there's a activation_name, there better be a function
    if activation_name is not None:
        return ACTIVATION_FROM_NAME[activation_name]


LayerResult = namedtuple('LayerResult', ['layer_info', 'ongoing_result'])


def build_features(
    network_config,
    input_placeholder,
    targets_placeholder,
    is_training,
    data_splits,
    reuse=True,
    ):
    with tf.variable_scope(network_config.config_id, reuse=reuse):
        layer_features = []

        ongoing_result = tf.cond(
            is_training,
            true_fn=lambda: network_config.data_augmentation_func(input_placeholder, is_training=True),
            false_fn=lambda: network_config.data_augmentation_func(input_placeholder, is_training=False),
            name='augment_data',
        )

        for i, layer in enumerate(network_config.layers):
            if not reuse: print(layer, ongoing_result)

            # This is the function that does a lot of work!
            # This takes in the list of layer features so far, and the current result.
            # This is a little weird, but it's so the input layer isn't added to our layer_features
            # and doesn't mess up indexing.
            layer_result = layer.compute(i, layer_features, ongoing_result, targets_placeholder, is_training)
            ongoing_result = layer_result.ongoing_result

            layer_features.append(layer_result)

            if not reuse:
                # maybe assert output shape if we're initially building it
                expected_output_shape = getattr(layer, 'assert_output_shape')
                if expected_output_shape is not None:
                    full_expected_output_shape = [network_config.batch_size] + expected_output_shape
                    print(full_expected_output_shape)
                    actual_output_shape = ongoing_result.get_shape().as_list()

                    assert actual_output_shape == full_expected_output_shape, 'Wrong shape! Ignoring batch size: actual {} expected {}'.format(
                        actual_output_shape,
                        full_expected_output_shape,
                    )

    return layer_features, ongoing_result


def get_result_of_layer_id(other_input_id, result_list):
    return result_list[other_input_id].ongoing_result


def layer_idx_with_id(layers, find_id):
    '''given a list of abcLayerModels (configs, not tensorflow), find the layer with the given find_id'''
    idx = None
    for i, layer in enumerate(layers):
        if layer.id == find_id:
            if idx is not None:
                raise Exception('multiple ids! {}'.format(find_id))
            idx = i
    else:
        if idx is None:
            raise Exception('prediction id `{}` isn\'t the id of a layer'.format(find_id))
    return idx


def slow_get_result_of_layer_id(layer_features, layer_id):
    '''Searches through the layer_features to find the layer_id.

    NOTE! This is mostly convenient for jupyter notebooks. If my mental model of tensorflow
    is right, this shouldn't be within in LayerModels, because it needs to iterate over all of the
    layers to get the index. The faster combination of suffixing a layer reference with
    "layer_id" and then using get_result_of_layer_id is a better idea!
    '''

    return layer_features[
        layer_idx_with_id(
            [x.layer_info for x in layer_features],
            layer_id
        )
    ].ongoing_result


## Begin Layer Models


class BaseLayerModel(object):
    optional_fields = DEFAULT_LAYER_RESULT_FIELDS

    def _compute(self, depth, result_list, ongoing_result, targets, is_training):
        raise NotImplementedError()

    def compute(self, depth, result_list, ongoing_result, targets, is_training):

        # if an input id is provided, use that instead of the previous layer's output
        if self.input_id is not None:
            ongoing_result = get_result_of_layer_id(self.input_id, result_list)

        with tf.variable_scope('layer_{}__{}'.format(depth, type(self).__name__)):
            result = self._compute(
                depth,
                result_list,
                ongoing_result,
                targets,
                is_training
            )

        return LayerResult(self, result)


class DenseLayer(BaseLayerModel, namedtuple('DenseLayer', DEFAULT_LAYER_RESULT_FIELDS + [
    'units',
    'activation',
])):
    def _compute(self, depth, result_list, ongoing_result, targets, is_training):
        ongoing_result = tf.layers.dense(
            ongoing_result,
            units=self.units)
        ongoing_result = ACTIVATION_FROM_NAME[self.activation](
            ongoing_result,
            name="{}{}".format(self.activation, depth)
        )
        return ongoing_result


class MaxPoolLayer(BaseLayerModel, namedtuple('MaxPoolLayer', DEFAULT_LAYER_RESULT_FIELDS + [
    'pool_size',
    'strides',
])):
    def _compute(self, depth, result_list, ongoing_result, targets, is_training):
        ongoing_result = tf.layers.max_pooling2d(
            ongoing_result,
            pool_size=self.pool_size,
            strides=self.strides,
        )
        return ongoing_result


class Conv2DLayer(BaseLayerModel, namedtuple('Conv2DLayer', DEFAULT_LAYER_RESULT_FIELDS + [
    'filter_count',
    'kernel_length',
    'strides',
    'padding',
    'activation',
])):
    optional_fields = OPTIONAL_DEFAULT_LAYER_RESULT_FIELDS + ['activation', 'padding']

    def _compute(self, depth, result_list, ongoing_result, targets, is_training):
        activation_func = activation_from_name(self.activation)

        padding = self.padding
        if self.padding is None:
            padding = 'valid'

        ongoing_result = tf.layers.conv2d(
            ongoing_result,
            self.filter_count,
            (self.kernel_length, self.kernel_length),
            strides=(self.strides, self.strides),
            padding=padding,
            activation=activation_func,
        )
        return ongoing_result


class DropoutLayer(BaseLayerModel, namedtuple('DropoutLayer', DEFAULT_LAYER_RESULT_FIELDS + [
    'rate',
])):
    def _compute(self, depth, result_list, ongoing_result, targets, is_training):
        ongoing_result = tf.layers.dropout(
            ongoing_result,
            rate=self.rate,
            training=is_training,
        )
        return ongoing_result


class BatchNormLayer(BaseLayerModel, namedtuple('BatchNormLayer', DEFAULT_LAYER_RESULT_FIELDS)):
    def _compute(self, depth, result_list, ongoing_result, targets, is_training):
        ongoing_result = batch_norm(
            ongoing_result,
            decay=0.99,
            scale=True,
            center=True,
            is_training=is_training,
            renorm=False
        )
        return ongoing_result


COMBINATION_METHOD_NAME_TO_FUNC = {
    'add_n': tf.add_n
}


class SkipConnectionLayer(BaseLayerModel, namedtuple('SkipConnectionLayer', DEFAULT_LAYER_RESULT_FIELDS + [
    'other_input_id',
    'combination_method',
])):
    '''Combines the input (`input_id`, or previous layer) with some other input (`other_input_id`) using the
    `combination_method` (see COMBINATION_METHOD_NAME_TO_FUNC).'''
    def _compute(self, depth, result_list, ongoing_result, targets, is_training):
        other_input = get_result_of_layer_id(self.other_input_id, result_list)

        ongoing_result = COMBINATION_METHOD_NAME_TO_FUNC[self.combination_method](
            [other_input, ongoing_result]
        )
        return ongoing_result


def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector


def caps_conv_helper(conv_layer, ongoing_result):
    activation_func = activation_from_name(conv_layer.activation)

    return tf.layers.conv2d(
        ongoing_result,
        conv_layer.caps_count * conv_layer.caps_dim,
        (conv_layer.kernel_length, conv_layer.kernel_length),
        strides=(conv_layer.strides, conv_layer.strides),
        padding='valid',
        activation=activation_func,
    )


def caps_squash_helper(squashing_layer, ongoing_result):

    batch_size, input_x, input_y, _ = ongoing_result.get_shape().as_list()
    map_count = input_x * input_y * squashing_layer.caps_count

    ongoing_result = tf.reshape(
        ongoing_result,
        [batch_size, map_count, squashing_layer.caps_dim],
        name="caps_raw"
    )

    ongoing_result = squash(
        ongoing_result,
        name='caps_squash'
    )

    return ongoing_result


# These are based on https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb
# TODO: maybe deprecate this and just use PrimaryCapsuleCube
class PrimaryCapsule(BaseLayerModel, namedtuple('PrimaryCapsule', DEFAULT_LAYER_RESULT_FIELDS + [
    'caps_count',
    'caps_dim',
    'kernel_length',
    'strides',
    'activation',
    ])):

    optional_fields = OPTIONAL_DEFAULT_LAYER_RESULT_FIELDS + ['activation']

    def _compute(self, depth, result_list, ongoing_result, targets, is_training):
        ongoing_result = caps_conv_helper(self, ongoing_result)
        ongoing_result = caps_squash_helper(self, ongoing_result)
        return ongoing_result


# This is like PrimaryCapsule, but the output is a cube. I'd like to deprecate PrimaryCapsule
class PrimaryCapsuleCube(BaseLayerModel, namedtuple('PrimaryCapsuleCube', DEFAULT_LAYER_RESULT_FIELDS + [
    'caps_count',
    'caps_dim',
    'kernel_length',
    'strides',
    'activation',
    ])):

    optional_fields = OPTIONAL_DEFAULT_LAYER_RESULT_FIELDS + ['activation']

    def _compute(self, depth, result_list, ongoing_result, targets, is_training):
        ongoing_result = caps_conv_helper(self, ongoing_result)

        # grab the convolution output size before we squash it
        batch_size, input_x, input_y, _ = ongoing_result.get_shape().as_list()

        ongoing_result = caps_squash_helper(self, ongoing_result)

        ongoing_result = tf.reshape(
            ongoing_result,
            [batch_size, input_x, input_y, self.caps_count * self.caps_dim],
            'caps_cube',
        )

        return ongoing_result


def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


# I'm not sure if this is the best way to break this up
# but this is called a "DigitCapsule" in the notebook
class OutCapsule(BaseLayerModel, namedtuple('OutCapsule', DEFAULT_LAYER_RESULT_FIELDS + [
    'caps_count',
    'caps_dim',
    ])):
    def _compute(self, depth, result_list, ongoing_result, targets, is_training):

        batch_size, input_caps_count, input_caps_dim = ongoing_result.get_shape().as_list()

        init_sigma = 0.01

        # TODO: double check the below variables are scoped properly
        W_init = tf.random_normal(
            shape=(
                1,
                input_caps_count,
                self.caps_count,
                self.caps_dim,
                input_caps_dim,
            ),
            stddev=init_sigma,
            dtype=tf.float32,
            name="W_init")

        W = tf.get_variable("W", initializer=W_init)

        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

        caps2_n_caps = self.caps_count
        caps1_output = ongoing_result

        caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                               name="caps1_output_expanded")
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                           name="caps1_output_tile")
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                                     name="caps1_output_tiled")

        ongoing_result = tf.matmul(
            W_tiled,
            caps1_output_tiled,
            name="caps2_predicted"
        )

        return ongoing_result


class DynamicRouting(BaseLayerModel, namedtuple('DynamicRouting', DEFAULT_LAYER_RESULT_FIELDS + [
    'iterations',
    ])):
    def _compute(self, depth, result_list, ongoing_result, targets, is_training):
        # TODO: make this a variable
        assert self.iterations == 2, 'this only support caps_routing with iterations=2 atm'

        caps2_predicted = ongoing_result
        batch_size, input_caps_count, output_caps_count, __, ___ = caps2_predicted.get_shape().as_list()

        caps1_n_caps = input_caps_count
        caps2_n_caps = output_caps_count

        raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                   dtype=np.float32, name="raw_weights")

        routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

        weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                           name="weighted_predictions")
        weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                                     name="weighted_sum")

        caps2_output_round_1 = squash(weighted_sum, axis=-2,
                                      name="caps2_output_round_1")

        caps2_output_round_1_tiled = tf.tile(
            caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
            name="caps2_output_round_1_tiled")

        agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                              transpose_a=True, name="agreement")

        raw_weights_round_2 = tf.add(raw_weights, agreement,
                                     name="raw_weights_round_2")

        routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                                dim=2,
                                                name="routing_weights_round_2")
        weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                                   caps2_predicted,
                                                   name="weighted_predictions_round_2")
        weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                             axis=1, keep_dims=True,
                                             name="weighted_sum_round_2")
        caps2_output_round_2 = squash(weighted_sum_round_2,
                                      axis=-2,
                                      name="caps2_output_round_2")

        ongoing_result = caps2_output_round_2

        return ongoing_result


class CapsulePrediction(BaseLayerModel, namedtuple('CapsulePrediction', DEFAULT_LAYER_RESULT_FIELDS)):
    def _compute(self, depth, result_list, ongoing_result, targets, is_training):
        caps2_output = ongoing_result
        with tf.variable_scope('predicting'):
            y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")
            y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba_max")
            y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2], name="y_pred")
        return y_pred


class CapsuleMasking(BaseLayerModel, namedtuple('CapsuleMasking', DEFAULT_LAYER_RESULT_FIELDS + [
    'routing_capsule_input_id'
    ])):
    def _compute(self, depth, result_list, ongoing_result, targets, is_training):

        y_pred = ongoing_result

        # dynamic routing requires both prediction (which should be in ongoing_result)
        # and the result of a single Out Capsule.
        caps2_output = get_result_of_layer_id(self.routing_capsule_input_id, result_list)

        batch_size, _, output_caps_count, output_caps_dim, ___ = caps2_output.get_shape().as_list()

        # During training, we should "send only the output vector of the capsule that
        # corresponds to the target digit".
        # So when training, mask on target, and when not training, mask on prediction.
        reconstruction_targets = tf.cond(
            is_training, # condition
            true_fn=lambda: tf.cast(targets, tf.int64),
            false_fn=lambda: y_pred,
            name="reconstruction_targets")

        reconstruction_mask = tf.one_hot(reconstruction_targets,
                         depth=output_caps_count,
                         name="reconstruction_mask")

        reconstruction_mask_reshaped = tf.reshape(
            reconstruction_mask, [-1, 1, output_caps_count, 1, 1],
            name="reconstruction_mask_reshaped")

        caps2_output_masked = tf.multiply(
            caps2_output, reconstruction_mask_reshaped,
            name="caps_output_masked")

        return caps2_output_masked


class SoftmaxPrediction(BaseLayerModel, namedtuple('SoftmaxPrediction', DEFAULT_LAYER_RESULT_FIELDS + ['num_classes'])):
    def _compute(self, depth, result_list, ongoing_result, targets, is_training):

        c_conv_encoder = ongoing_result
        c_conv_encoder = tf.contrib.layers.flatten(c_conv_encoder)
        c_conv_encoder = tf.layers.dense(
            c_conv_encoder,
            units=self.num_classes
        )
        ongoing_result = c_conv_encoder

        return ongoing_result


class ArgMaxLayer(BaseLayerModel, namedtuple('ArgMaxLayer', DEFAULT_LAYER_RESULT_FIELDS)):
    def _compute(self, depth, result_list, ongoing_result, targets, is_training):
        return tf.argmax(ongoing_result, 1)


class ReshapeLayer(BaseLayerModel, namedtuple('ReshapeLayer', DEFAULT_LAYER_RESULT_FIELDS + [
    'shape'
    ])):
    def _compute(self, depth, result_list, ongoing_result, targets, is_training):
        return tf.reshape(ongoing_result, self.shape)


# key -> func(images, shape excluding batch/channels)
UPSCALE_METHODS = {
    'nearest_neighbors': tf.image.resize_nearest_neighbor
}


# Deconvolution stuff from https://github.com/AntreasAntoniou/DAGAN/blob/master/dagan_architectures.py
class UpscaleLayer(BaseLayerModel, namedtuple('UpscaleLayer', DEFAULT_LAYER_RESULT_FIELDS + [
    'shape',
    'method',
    ])):
    def _compute(self, depth, result_list, ongoing_result, targets, is_training):
        return UPSCALE_METHODS[self.method](ongoing_result, self.shape)


# Deconvolution stuff from https://github.com/AntreasAntoniou/DAGAN/blob/master/dagan_architectures.py
class DeconvLayer(BaseLayerModel, namedtuple('DeConvLayer', DEFAULT_LAYER_RESULT_FIELDS + [
    'filter_size',
    'filter_count',
    'activation',
    ])):

    optional_fields = OPTIONAL_DEFAULT_LAYER_RESULT_FIELDS + ['activation']

    def _compute(self, depth, result_list, ongoing_result, targets, is_training):

        activation_func = activation_from_name(self.activation)

        return tf.layers.conv2d_transpose(
            ongoing_result,
            self.filter_count,
            (self.filter_size, self.filter_size),
            strides=(1, 1), # for now, use strides of size 1
            padding="SAME",
            activation=activation_func,
        )


LAYER_TYPE_TO_NAMEDTUPLE = {
    'dense': DenseLayer,
    'batch_norm': BatchNormLayer,
    'max_pool': MaxPoolLayer,
    'conv': Conv2DLayer,
    'dropout': DropoutLayer,
    'caps_primary': PrimaryCapsule,
    'caps_primary_cube': PrimaryCapsuleCube,
    'caps_routing': DynamicRouting,
    'caps_out': OutCapsule,
    'caps_pred': CapsulePrediction,
    'caps_masking': CapsuleMasking,
    'softmax_pred': SoftmaxPrediction,
    'argmax': ArgMaxLayer,
    'reshape': ReshapeLayer,
    'skip_connection': SkipConnectionLayer,
    'upscale': UpscaleLayer,
    'deconv': DeconvLayer,
}


def layer_from_dict(layer_dict):
    layer_type = layer_dict.pop('type')

    layer_model_class = LAYER_TYPE_TO_NAMEDTUPLE[layer_type]

    # populate optional fields
    for field in layer_model_class.optional_fields:
        if field not in layer_dict:
            layer_dict[field] = None

    return layer_model_class(**layer_dict)


def layers_from_list(layers_dict_list):

    raw_layers = [
        layer_from_dict(layer_dict)
        for layer_dict in layers_dict_list
    ]
    # ew. replace ids with indices, but don't update the name gross

    # if this namedtuple has a something_input_id field, replace the ids with indices
    layers = []
    for layer in raw_layers:
        update_these = {}
        for field in layer._fields:
            if field.endswith('input_id') and getattr(layer, field, None) is not None:
                update_these[field] = layer_idx_with_id(raw_layers, getattr(layer, field))
        layer = layer._replace(**update_these)
        layers.append(layer)

    return layers
