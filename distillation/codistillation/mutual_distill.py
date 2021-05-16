"""
    Co-distillation training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from absl import flags
import numpy as np
from nets import nets_factory
from utils import tiny_imagenet_dataset_creator

FLAGS = flags.FLAGS


def kl_loss_compute(logits1, logits2):
    """
    From logits1 to logits2, logits2 is the "truth", whereas logits1 is the "pred"
    """
    pred1 = tf.nn.softmax(logits1)
    pred2 = tf.nn.softmax(logits2)
    loss = tf.reduce_mean(tf.reduce_sum(pred2 * tf.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))
    return loss


def create_distill_graph(network_fn, images, labels):
    """Calculate the total loss on a single tower running the reid model."""
    # Build inference Graph.
    net_endpoints, net_logits, net_raw_loss = {}, {}, {}  # temporary usage
    net_pred = {}
    for i in range(FLAGS.num_networks):
        net_logits["{0}".format(i)], net_endpoints["{0}".format(i)] = \
            network_fn["{0}".format(i)](images, scope=('dmlnet_%d' % i))
        net_raw_loss["{0}".format(i)] = tf.losses.softmax_cross_entropy(
                logits=net_logits["{0}".format(i)], onehot_labels=labels,
                label_smoothing=FLAGS.label_smoothing, weights=1.0)
        net_pred["{0}".format(i)] = net_endpoints["{0}".format(i)]['Predictions']

    # Add KL loss if there are more than one network
    net_loss, kl_loss, peer_logits, net_reg_loss, net_loss_averages, net_loss_averages_op = {}, {}, {}, {}, {}, {}
    net_total_loss = {}

    for i in range(FLAGS.num_networks):
        net_loss["{0}".format(i)] = net_raw_loss["{0}".format(i)]
        
        # Calculate peer ensembled kl loss
        peer_logits["{0}".format(i)] = tf.add_n([net_logits["{0}".format(j)] for j in range(FLAGS.num_networks)
                                                 if j != i]) / float(FLAGS.num_networks - 1)
        kl_loss["{0}".format(i)] = kl_loss_compute(net_logits["{0}".format(i)], peer_logits["{0}".format(i)])
        tf.summary.scalar('kl_loss_%d' % i, kl_loss["{0}".format(i)])
        
        # Total backprop loss term: kl_loss * weight + normal_loss
        net_loss["{0}".format(i)] += kl_loss["{0}".format(i)] * FLAGS.kl_loss_weight
        # Regularization loss terms: a list of tensors
        net_reg_loss["{0}".format(i)] = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=('dmlnet_%d' % i))

        # Get total loss for net_i: reg_loss + kl_loss + cross_entropy_loss
        net_total_loss["{0}".format(i)] = tf.add_n([net_loss["{0}".format(i)]] +
                                                   net_reg_loss["{0}".format(i)],
                                                   name=('net%d_total_loss' % i))

        tf.summary.scalar('net%d_loss_raw' % i, net_raw_loss["{0}".format(i)])
        tf.summary.scalar('net%d_loss_total' % i, net_total_loss["{0}".format(i)])

    return net_total_loss, net_pred, net_raw_loss


def train():
    # if not FLAGS.dataset_dir:
    #     raise ValueError('You must supply the dataset directory with --dataset_dir')
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        ######################
        # Create the session #
        ######################
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        ######################
        # Select the network #
        ######################
        network_fn = {}
        model_names = [net.strip() for net in FLAGS.model_name.split(',')]
        for i in range(FLAGS.num_networks):
            network_fn["{0}".format(i)] = nets_factory.get_network_fn(
                model_names[i],
                num_classes=FLAGS.num_classes,
                weight_decay=FLAGS.weight_decay,
                is_training=True)

        #########################################
        # Configure the optimization procedure. #
        #########################################
        raw_opt, net_opt = {}, {}
        for i in range(FLAGS.num_networks):
            net_opt["{0}".format(i)] = tf.train.AdamOptimizer(FLAGS.learning_rate,
                                                              beta1=FLAGS.adam_beta1,
                                                              beta2=FLAGS.adam_beta2,
                                                              epsilon=FLAGS.opt_epsilon)
            raw_opt["{0}".format(i)] = tf.train.AdamOptimizer(FLAGS.learning_rate,
                                                              beta1=FLAGS.adam_beta1,
                                                              beta2=FLAGS.adam_beta2,
                                                              epsilon=FLAGS.opt_epsilon)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        with tf.device("/cpu:0"):
            train_dataset = tiny_imagenet_dataset_creator\
                .create_tiny_imagenet_input_dataset('validation', FLAGS.batch_size, FLAGS.image_size, is_training=True)
            validation_dataset = tiny_imagenet_dataset_creator\
                .create_tiny_imagenet_input_dataset('validation', FLAGS.batch_size, FLAGS.image_size, False)
            handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(handle,
                                                           train_dataset.output_types, train_dataset.output_shapes)
            images, labels = iterator.get_next()
            train_iter = train_dataset.make_one_shot_iterator()
            train_handle = sess.run(train_iter.string_handle())
            validation_iter = validation_dataset.make_initializable_iterator()
            validation_handle = sess.run(validation_iter.string_handle())

            # dataset_iterator = train_dataset.make_one_shot_iterator()
            # images, labels = dataset_iterator.get_next()
            # labels = tf.one_hot(labels, FLAGS.num_classes)

        ##############################################################
        # Construct the actual training op #
        ##############################################################
        raw_stats, stats, precision, net_var_list, raw_grads, net_grads = {}, {}, {}, {}, {}, {}
        truth = tf.argmax(labels, axis=1)

        # Construct the overall co-distill graph
        net_loss, net_pred, raw_loss = create_distill_graph(network_fn, images, labels)
        # Retain the summary.
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        var_list = tf.trainable_variables()

        for i in range(FLAGS.num_networks):
            predictions = tf.argmax(net_pred["{0}".format(i)], axis=1)
            precision["{0}".format(i)] = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))
            # Add a summary to track the training precision for each network.
            summaries.append(tf.summary.scalar('precision_%d' % i, precision["{0}".format(i)]))

            stats["{0}".format(i)] = tf.stack([net_loss["{0}".format(i)], precision["{0}".format(i)]])
            raw_stats["{0}".format(i)] = tf.stack([raw_loss["{0}".format(i)], precision["{0}".format(i)]])

            # Generate gradient calculation op w.r.t. each specific network
            net_var_list["{0}".format(i)] = \
                [var for var in var_list if 'dmlnet_%d' % i in var.name]
            net_grads["{0}".format(i)] = net_opt["{0}".format(i)].compute_gradients(
                net_loss["{0}".format(i)], var_list=net_var_list["{0}".format(i)])
            raw_grads["{0}".format(i)] = raw_opt["{0}".format(i)].compute_gradients(
                raw_loss["{0}".format(i)], var_list=net_var_list["{0}".format(i)])

        # # Add histograms for histogram and trainable variables.
        # for i in range(FLAGS.num_networks):
        #     for grad, var in net_grads["{0}".format(i)]:
        #         if grad is not None:
        #             summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        # for var in tf.trainable_variables():
        #     summaries.append(tf.summary.histogram(var.op.name, var))

        # Apply the gradients to construct the train_op.
        raw_train_op, net_train_op = {}, {}
        for i in range(FLAGS.num_networks):
            net_train_op["{0}".format(i)] = net_opt["{0}".format(i)].apply_gradients(net_grads["{0}".format(i)])
            raw_train_op["{0}".format(i)] = raw_opt["{0}".format(i)].apply_gradients(raw_grads["{0}".format(i)])

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        #############################################
        # Create session and start training #
        #############################################
        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess.run(init)
        sess.run(validation_iter.initializer)

        summary_writer = tf.summary.FileWriter(
            os.path.join(FLAGS.tb_log_dir),
            graph=sess.graph)

        # These values are training loss w.r.t. a successive batch
        stats_value, net_loss_value, precision_value = {}, {}, {}

        # In total # epoches.
        for e in range(FLAGS.epoches):

            for step in range(FLAGS.max_number_of_train_steps):
                for i in range(FLAGS.num_networks):
                    _, stats_value["{0}".format(i)] = sess.run([raw_train_op["{0}".format(i)], raw_stats["{0}".format(i)]], feed_dict={handle: train_handle})
                    net_loss_value["{0}".format(i)] = stats_value["{0}".format(i)][0]
                    precision_value["{0}".format(i)] = stats_value["{0}".format(i)][1]
                    assert not np.isnan(net_loss_value["{0}".format(i)]), 'Model diverged with loss = NaN'

            for step in range(FLAGS.max_number_of_codistill_steps):
                for i in range(FLAGS.num_networks):
                    _, stats_value["{0}".format(i)] = sess.run([net_train_op["{0}".format(i)], stats["{0}".format(i)]], feed_dict={handle: train_handle})
                    net_loss_value["{0}".format(i)] = stats_value["{0}".format(i)][0]
                    precision_value["{0}".format(i)] = stats_value["{0}".format(i)][1]
                    assert not np.isnan(net_loss_value["{0}".format(i)]), 'Model diverged with loss = NaN'

                if step % 100 == 0:
                    summary_str = sess.run(summary_op, feed_dict={handle: train_handle})
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step % FLAGS.ckpt_steps == 0 or (step + 1) == FLAGS.max_number_of_steps:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

                # periodically print training/validation stats to screen
                if step % 1000 == 0:
                    format_str = 'Epoch {0}, step {1}, net0_loss = {2:.4f}, net0_acc = {2:.6f}'
                    print(format_str.format(e, step, net_loss_value["{0}".format(0)], precision_value["{0}".format(0)]))
                    sess.run(validation_iter.initializer)
                    prec, los = 0.0, 0.0
                    for _ in range(5):
                        stat_val = sess.run(stats["{0}".format(0)], feed_dict={handle: validation_handle})
                        los, prec = los + stat_val[0], prec + stat_val[1]
                    los, prec = los / 5, prec / 5
                    print('Epoch {0}, Step {1}, Validation loss {2:.4f}, precision {3:.6f}'.format(e, step, los, prec))


