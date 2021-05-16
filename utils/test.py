import tensorflow as tf
import tensorflow.python.keras as tfkeras
from utils import tiny_imagenet_dataset_creator

from absl import app
from absl import flags

config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
config.log_device_placement = False


sess = tf.Session(config=config)
tfkeras.backend.set_session(sess)

flags.DEFINE_string('tiny_imagenet_data_dir', '..',
                    'Directory with Tiny Imagenet dataset in TFRecord format.')


def train(_):
    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    train_dataset = tiny_imagenet_dataset_creator \
        .create_tiny_imagenet_input_dataset('train', batch_size=32, image_size=128, is_training=True)
    validation_dataset = tiny_imagenet_dataset_creator \
        .create_tiny_imagenet_input_dataset('validation', batch_size=100, image_size=128, is_training=True)
    dataset_iterator = train_dataset.make_one_shot_iterator()
    valid_iterator = validation_dataset.make_one_shot_iterator()
    images, labels = dataset_iterator.get_next()
    valid_imgs, valid_labs = valid_iterator.get_next()
    # labels = tf.one_hot(labels, 200)
    # valid_labs = tf.one_hot(valid_labs, 200)

    net = tfkeras.applications.mobilenet_v2.MobileNetV2(weights=None, classes=200)
    preds = net(images)
    loss = tf.reduce_mean(tfkeras.losses.categorical_crossentropy(labels, preds))
    train_op = tf.train.AdamOptimizer().minimize(loss)

    valid_preds = net(valid_imgs)
    validation_op = tf.reduce_mean(tfkeras.metrics.categorical_accuracy(valid_labs, valid_preds))

    init_op = tf.global_variables_initializer()

    with sess.as_default():
        sess.run(init_op)
        for i in range(200000):
            sess.run(train_op)
            if i % 200 == 0:
                accy = sess.run(validation_op)
                print('validation accuracy for step {0}: {1}'.format(i, accy))


if __name__ == "__main__":
    app.run(train)
