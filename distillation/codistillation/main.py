"""
    Entry point of co-distillation training
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distillation.codistillation.mutual_distill as mutual_distill
from absl import app
from absl import flags

#########################
# Training Directories #
#########################
flags.DEFINE_string('tiny_imagenet_data_dir', 'data',
                    'Directory with Tiny Imagenet dataset in TFRecord format.')

flags.DEFINE_integer('image_size', 128,
                     'The number of samples in each batch.')

flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                    'Directory name to save the checkpoints [checkpoint]')

flags.DEFINE_string('tb_log_dir', 'logs',
                    'Directory name to save the tensorboard logs')

#########################
#     Model Settings    #
#########################

flags.DEFINE_string('model_name', 'mobilenet_v1, mobilenet_v1',
                    'The name of the architecture to train.')

flags.DEFINE_float('weight_decay', 0.00004,
                   'The weight decay on the model weights.')

flags.DEFINE_float('label_smoothing', 0.0,
                   'The amount of label smoothing.')

flags.DEFINE_integer('batch_size', 16,
                     'The number of samples in each batch.')

flags.DEFINE_integer('max_number_of_codistill_steps', 6000,    # max total steps 6000 * 10, previous total steps 200000
                     'The maximum number of training steps.')

flags.DEFINE_integer('max_number_of_train_steps', 6000,
                     'The maximum number of training steps.')

flags.DEFINE_integer('epoches', 10,
                     'The maximum number of training steps.')

flags.DEFINE_integer('ckpt_steps', 5000,
                     'How many steps to save checkpoints.')

flags.DEFINE_integer('num_classes', 200,
                     'The number of classes.')

flags.DEFINE_integer('num_networks', 2,
                     'The number of networks in DML.')

#########################
# Optimization Settings #
#########################

flags.DEFINE_string('optimizer', 'adam',
                    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
                    '"ftrl", "momentum", "sgd" or "rmsprop".')

flags.DEFINE_float('learning_rate', 0.001,
                   'Initial learning rate.')

flags.DEFINE_float('adam_beta1', 0.9,
                   'The exponential decay rate for the 1st moment estimates.')

flags.DEFINE_float('adam_beta2', 0.999,
                   'The exponential decay rate for the 2nd moment estimates.')

flags.DEFINE_float('opt_epsilon', 1e-8,
                   'Epsilon term for the optimizer.')

#########################
#   Co-distill Settings #
#########################
flags.DEFINE_float('kl_loss_weight', 1.0,
                   'KL loss term weight when calculating total loss')

#########################
#      Misc Settings    #
#########################
flags.DEFINE_boolean('log_device_placement', False,
                     """Whether to log device placement.""")

FLAGS = flags.FLAGS


def main(_):
    mutual_distill.train()


if __name__ == '__main__':
    app.run(main)
