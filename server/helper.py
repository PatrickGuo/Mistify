"""
Utility functions needed for Adaptation server
"""
import logging
import os
import uuid

import zmq


def set_logger(context, verbose=False):
    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s',
        datefmt='%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


def check_tf_version():
    import tensorflow as tf
    tf_ver = tf.__version__.split('.')
    return tf_ver


def import_tf(device_id=-1, verbose=False, use_fp16=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
    os.environ['TF_FP16_MATMUL_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    os.environ['TF_FP16_CONV_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG if verbose else tf.compat.v1.logging.ERROR)
    return tf


def auto_bind(socket):
    """
    Helper function to automatically bind socket to a local random directory for ipc.
    :param socket:
    :return:
    """
    # Get the location for tmp file for sockets
    try:
        tmp_dir = os.environ['ZEROMQ_SOCK_TMP_DIR']
        if not os.path.exists(tmp_dir):
            raise ValueError('This directory for sockets ({}) does not seems to exist.'.format(tmp_dir))
        tmp_dir = os.path.join(tmp_dir, str(uuid.uuid1())[:8])
    except KeyError:
        tmp_dir = '*'

    socket.bind('ipc://{}'.format(tmp_dir))
    return socket.getsockopt(zmq.LAST_ENDPOINT).decode('ascii')
