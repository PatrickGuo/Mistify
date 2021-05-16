import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import tensorflow.python.keras as tfkeras
from utils import tiny_imagenet_dataset_creator

from absl import app
from absl import flags

tf.enable_eager_execution()

# config = tf.ConfigProto()
# config.allow_soft_placement = True
# config.gpu_options.allow_growth = True
# config.log_device_placement = False
#
#
# sess = tf.Session(config=config)
# tfkeras.backend.set_session(sess)

flags.DEFINE_string('tiny_imagenet_data_dir', '..',
                    'Directory with Tiny Imagenet dataset in TFRecord format.')

EPOCHS = 12


def main(_):
    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    train_dataset = tiny_imagenet_dataset_creator \
        .create_tiny_imagenet_input_dataset('train', batch_size=32, image_size=128, is_training=True)
    validation_dataset = tiny_imagenet_dataset_creator \
        .create_tiny_imagenet_input_dataset('validation', batch_size=50, image_size=128, is_training=True)

    model = tfkeras.applications.mobilenet_v2.MobileNetV2(weights=None, classes=200)

    optimizer = tf.train.AdamOptimizer()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = tf.reduce_mean(tfkeras.losses.categorical_crossentropy(labels, predictions))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    def test_step(images, labels):
        predictions = model(images)
        t_loss = tf.reduce_mean(tfkeras.losses.categorical_crossentropy(labels, predictions))

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    for epoch in range(EPOCHS):
        step = 0
        for images, labels in train_dataset:
            labels = tf.one_hot(labels, 200)
            train_step(images, labels)
            if step % 100 == 0:
                template = 'Epoch {}, Step {}, Loss: {}, Accuracy: {}'
                print(template.format(epoch + 1,
                                      step,
                                      train_loss.result(),
                                      train_accuracy.result() * 100))
            step += 1

        for test_images, test_labels in validation_dataset:
            test_step(test_images, test_labels)

        templat1 = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(templat1.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))



    # net = tfkeras.applications.mobilenet_v2.MobileNetV2(weights=None, classes=200)
    # preds = net(images)
    # loss = tf.reduce_mean(tfkeras.losses.categorical_crossentropy(labels, preds))
    # train_op = tf.train.AdamOptimizer().minimize(loss)
    #
    # valid_preds = net(valid_imgs)
    # validation_op = tf.reduce_mean(tfkeras.metrics.categorical_accuracy(valid_labs, valid_preds))
    #
    # init_op = tf.global_variables_initializer()
    #
    # with sess.as_default():
    #     sess.run(init_op)
    #     for i in range(200000):
    #         sess.run(train_op)
    #         if i % 200 == 0:
    #             accy = sess.run(validation_op)
    #             print('validation accuracy for step {0}: {1}'.format(i, accy))


if __name__ == "__main__":
    app.run(main)
