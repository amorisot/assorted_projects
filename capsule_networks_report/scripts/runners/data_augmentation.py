import tensorflow as tf


DEFAULT_DATA_AUGMENTATION_NAME = 'na'


# Data Augmentation functions! This one is mlp's, copy-pasta'd from the mlpractical repo
def mlp_rotate_batch(batch_images, is_training):
    """
    Rotate a batch of images
    :param batch_images: A batch of images
    :param is_training: bool, whether this is training time
    :return: A rotated batch of images (some images will not be rotated if their rotation flip ends up False)
    """

    # If it's not training, just do a no-op
    if not is_training:
        return batch_images

    shapes = map(int, list(batch_images.get_shape()))
    if len(list(batch_images.get_shape())) < 4:
        raise Exception('You told me to augment the inputs but I can\'t because they don\'t have enough dimensions.')
    batch_size, x, y, c = shapes
    with tf.name_scope('augment'):
        batch_images_unpacked = tf.unstack(batch_images)
        new_images = []
        for image in batch_images_unpacked:
            new_images.append(mlp_rotate_image(image))
        new_images = tf.stack(new_images)
        new_images = tf.reshape(new_images, (batch_size, x, y, c))
        return new_images


def mlp_rotate_image(image):
    """
    Rotates a single image
    :param image: An image to rotate
    :return: A rotated or a non rotated image depending on the result of the flip
    """
    no_rotation_flip = tf.unstack(
        tf.random_uniform([1], minval=1, maxval=100, dtype=tf.int32, seed=None,
                          name=None))  # get a random number between 1 and 100
    flip_boolean = tf.less_equal(no_rotation_flip[0], 50)
    # if that number is less than or equal to 50 then set to true
    random_variable = tf.unstack(tf.random_uniform([1], minval=1, maxval=3, dtype=tf.int32, seed=None, name=None))
    # get a random variable between 1 and 3 for how many degrees the rotation will be i.e. k=1 means 1*90,
    # k=2 2*90 etc.
    image = tf.cond(flip_boolean, lambda: tf.image.rot90(image, k=random_variable[0]),
                    lambda: image)  # if flip_boolean is true the rotate if not then do not rotate
    return image


def data_augmentation_noop(inputs, is_training):
    return inputs


DATA_AUGMENTATION_FUNC_FROM_NAME = {
    DEFAULT_DATA_AUGMENTATION_NAME: data_augmentation_noop,
    'mlp': mlp_rotate_batch,
}
