"""Tiny imagenet input dataset creator.
More examples see https://github.com/tensorflow/models/tree/master/research/adversarial_logit_pairing/datasets
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from absl import app
import tensorflow as tf

FLAGS = flags.FLAGS


def tiny_imagenet_parser(value, image_size, is_training):
    """Parses a single tiny imagenet example.
    Args:
      value: encoded example.
      image_size: size of the image.
      is_training: if True then do training preprocessing (which includes
        random cropping), otherwise do eval preprocessing.
    Returns:
      image: tensor with the image.
      label: true label of the image.
    """
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'label/tiny_imagenet': tf.FixedLenFeature([], tf.int64, -1),
    }

    parsed = tf.parse_single_example(value, keys_to_features)

    image_buffer = tf.reshape(parsed['image/encoded'], shape=[])
    image = tf.image.decode_image(image_buffer, channels=3)
    image = tf.image.convert_image_dtype(
        image, dtype=tf.float32)

    # Crop image
    if is_training:
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0],
                                       dtype=tf.float32,
                                       shape=[1, 1, 4]),
            min_object_covered=0.5,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.5, 1.0],
            max_attempts=20,
            use_image_if_no_bounding_boxes=True)
        image = tf.slice(image, bbox_begin, bbox_size)

    # resize image
    image = tf.image.resize_bicubic([image], [image_size, image_size])[0]

    # Rescale image to [-1, 1] range.
    image = tf.multiply(tf.subtract(image, 0.5), 2.0)

    image = tf.reshape(image, [image_size, image_size, 3])

    # Labels are in [0, 199] range
    label = tf.cast(
        tf.reshape(parsed['label/tiny_imagenet'], shape=[]), dtype=tf.int32)

    return image, label


def create_tiny_imagenet_input_dataset(split, batch_size, image_size, is_training):
    """Returns Tiny Imagenet Dataset.
    Args:
      split: name of the split, "train" or "validation".
      batch_size: size of the minibatch.
      image_size: size of the one side of the image. Output images will be
        resized to square shape image_size*image_size.
      is_training: if True then training preprocessing is done, otherwise eval
        preprocessing is done.instance of tf.data.Dataset with the dataset.
    Raises:
      ValueError: if name of the split is incorrect.
    Returns:
      Instance of tf.data.Dataset with the dataset.
    """
    assert FLAGS.tiny_imagenet_data_dir, 'TFRecord directory must be provided'
    if split.lower().startswith('train'):
        filepath = os.path.join(FLAGS.tiny_imagenet_data_dir, 'train.tfrecord')
    elif split.lower().startswith('validation'):
        filepath = os.path.join(FLAGS.tiny_imagenet_data_dir, 'validation.tfrecord')
    else:
        raise ValueError('Invalid split: %s' % split)

    dataset = tf.data.TFRecordDataset(filepath, buffer_size=8 * 1024 * 1024)

    if is_training:
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat()

    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda value: tiny_imagenet_parser(value, image_size, is_training),
            batch_size=batch_size,
            num_parallel_batches=4,
            drop_remainder=True))

    def set_shapes(images, labels):
        """Statically set the batch_size dimension."""
        images.set_shape(images.get_shape().merge_with(
            tf.TensorShape([batch_size, None, None, None])))
        labels.set_shape(labels.get_shape().merge_with(
            tf.TensorShape([batch_size])))
        return images, labels

    # Assign static batch size dimension
    dataset = dataset.map(set_shapes)

    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


def main(_):
    """
    Test whether function works correctly
    """
    dataset = create_tiny_imagenet_input_dataset('validation', 2, 128, True)
    dataset_iterator = dataset.make_one_shot_iterator()
    images, labels = dataset_iterator.get_next()
    with tf.Session() as sess:
        im, lab = sess.run([images, labels])
        print(im)
        print(lab)


if __name__ == '__main__':
    app.run(main)

