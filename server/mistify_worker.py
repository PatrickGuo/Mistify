"""
AdaptWorker:
polling adaptation requests from the frontend, initiate and run the adaptExec state-machine.
"""

import os
import argparse
import multiprocessing
from multiprocessing import Process
from termcolor import colored
import tensorflow as tf
import zmq
import zmq.decorators as zmqd
from zmq.utils import jsonapi

from adapt_utils import train_epoch, validate_epoch
from adapt_exec import AdaptExec
from helper import *
from zmq_decor import multi_sockets


class ServerCmd:
    terminate = b'TERMINATION'
    show_config = b'SHOW_CONFIG'
    new_job = b'REGISTER'
    adapted_path = b'ADAPTED_PATH'


class AdaptWorker(Process):
    def __init__(self, id, worker_address_list, sink_address, device_id):
        super().__init__()
        self.worker_id = id
        self.device_id = device_id
        self.logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), verbose=False)
        self.daemon = True
        self.exit_flag = multiprocessing.Event()
        self.worker_address = worker_address_list
        self.num_concurrent_socket = len(self.worker_address)
        self.sink_address = sink_address
        self.verbose = False

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def run(self):
        self._run()

    @zmqd.socket(zmq.PUSH)
    @multi_sockets(zmq.PULL, num_socket='num_concurrent_socket')
    def _run(self, sink, *receivers):
        self.logger.info('use device %s, load graph from %s' % ('cpu' if self.device_id < 0 else ('gpu: %d' % self.device_id), self.graph_path))

        # Each worker see all new jobs posted to server and ``compete'' by polling
        for sock, addr in zip(receivers, self.worker_address):
            sock.connect(addr)
        # Connect to sink address
        sink.connect(self.sink_address)
        poller = zmq.Poller()
        for rcv_sock in receivers:
            poller.register(rcv_sock, zmq.POLLIN)
        self.logger.info('ready and listening!')

        while not self.exit_flag.is_set():
            events = dict(poller.poll())  # at here we poll to ``compete for'' new jobs.
            for sock_idx, sock in enumerate(receivers):
                if sock in events:
                    job_id, conf_id, raw_msg = sock.recv_multipart()
                    msg = jsonapi.loads(raw_msg) # msg contains the full adapt_param dict.
                    client_id, req_id = job_id.split(b'#')
                    self.logger.info('new job\tsocket: %d\tsize: %d\tjobId: %s\tconfId: %s\tclient: %s'
                                     % (sock_idx, len(msg), job_id, conf_id, client_id))
                    total_iterations, adapted_path, snapshot_path = parse_args_and_run(job_id, conf_id, msg)
                    self.logger.info('finished adaptation job with %d iterations' % total_iterations)
                    sink.send_multipart([job_id, conf_id, adapted_path, snapshot_path])
                    self.logger.info('job done\tclient: %s' % client_id)


def parse_args_and_run(job_id, conf_id, msg_dict):
    num_epochs_default = 1000
    num_classes_default = 10
    batch_size_default = 1024
    base_model_name_default = "ResNet50"
    learning_rate_default = 0.0001
    morphnet_regularizer_algorithm_choices = ["GroupLasso", "Gamma"]
    morphnet_regularizer_algorithm_default = morphnet_regularizer_algorithm_choices[0]
    morphnet_target_cost_choices = ["FLOPs", "Latency", "ModelSize"]
    morphnet_target_costs_default = morphnet_target_cost_choices[0]
    morphnet_target_cost_thresholds_default = "1.0"
    morphnet_hardware_default = "V100"
    morphnet_regularizer_threshold_default = 1e-2
    morphnet_regularization_multiplier_default = 1000.0
    log_dir_default = "./morphnet_log"
    main_train_device_default = "/cpu:0"
    main_eval_device_default = "/gpu:0"
    num_cuda_device_default = 4
    random_seed_default = 0
    base_model_choices = [
        "ResNet50", "ResNet101", "ResNet152", "ResNet50V2", "ResNet101V2",
        "ResNet101V2", "ResNet152V2", "VGG16", "VGG19", "Xception",
        "InceptionV3", "InceptionResNetV2", "MobileNet", "MobileNetV2",
        "DenseNet121", "DenseNet169", "DenseNet201", "NASNetLarge",
        "NASNetMobile"
    ]
    num_epochs = msg_dict.get('num_epoch', num_epochs_default)
    num_classes = msg_dict.get('num_classes', num_classes_default)
    batch_size = msg_dict.get('batch_size', batch_size_default)
    base_model_name = msg_dict.get('base_model_name', base_model_name_default)
    base_model_path = msg_dict.get('base_path', '')
    learning_rate = msg_dict.get('learning_rate_name', learning_rate_default)
    morphnet_regularizer_algorithm = msg_dict.get('morphnet_regularizer_algorithm', morphnet_regularizer_algorithm_default)
    morphnet_target_costs = msg_dict.get('morphnet_target_costs', morphnet_target_costs_default).split('+')
    morphnet_target_cost_thresholds = [float(s) for s in msg_dict.get('morphnet_target_cost_thresholds', morphnet_target_cost_thresholds_default).split('+')]
    morphnet_hardware = msg_dict.get('morphnet_hardware', morphnet_hardware_default)
    morphnet_regularizer_threshold = msg_dict.get('morphnet_regularizer_threshold', morphnet_regularizer_threshold_default)
    morphnet_regularization_multiplier = msg_dict.get('morphnet_regularization_multiplier', morphnet_regularization_multiplier_default)
    log_dir = msg_dict.get('log_dir', log_dir_default)
    num_cuda_device = msg_dict.get('num_cuda_device', num_cuda_device_default)
    random_seed = msg_dict.get('random_seed', random_seed_default)
    main_train_device = msg_dict.get('main_train_device', main_train_device_default)
    main_eval_device = msg_dict.get('main_eval_device', main_eval_device_default)

    # Load cifar10 dataset
    (x_train, y_train), (x_valid,
                         y_valid) = tf.keras.datasets.cifar10.load_data()
    # Convert class vectors to binary class matrices.
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
    y_valid_onehot = tf.keras.utils.to_categorical(y_valid, num_classes)
    image_shape = x_train[1:]
    # Normalize image inputs
    x_train = x_train.astype("float32") / 255.0
    x_valid = x_valid.astype("float32") / 255.0

    # Initiate an Adaptation Executor
    if 'cost_threshold' not in msg_dict:
        msg_dict['cost_threshold'] = 1e10
    morphnet_regularization_strength_dummy = 1e-9
    adaptor = AdaptExec(
        job_id=job_id,
        conf_id=conf_id,
        base_model_name=base_model_name,
        base_model_path=base_model_path,
        num_classes=num_classes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_gpus=num_cuda_device,
        main_train_device=main_train_device,
        main_eval_device=main_eval_device,
        morphnet_regularizer_algorithm=morphnet_regularizer_algorithm,
        morphnet_target_costs=morphnet_target_costs,
        morphnet_target_cost_thresholds=morphnet_target_cost_thresholds,
        morphnet_hardware=morphnet_hardware,
        morphnet_regularizer_threshold=morphnet_regularizer_threshold,
        morphnet_regularization_strength=morphnet_regularization_strength_dummy,
        log_dir=log_dir)

    # Export the unmodified model configures.
    initial_costs = adaptor.measure_costs(inputs=x_train[:batch_size])
    print("*" * 100)
    print("Initial Model Cost: ", *initial_costs)
    morphnet_regularization_strength = 1.0 / initial_costs[0] * morphnet_regularization_multiplier
    print("Use Regularization Strength: {}".format(
        morphnet_regularization_strength))
    adaptor.adjust_params(x_train[:batch_size], morphnet_target_costs, initial_costs)
    print("*" * 100)
    # Export the unmodified model configures.
    adapt_path = adaptor.export_model_config_with_inputs(inputs=x_train[:batch_size])
    # Iterative structure learning.
    for epoch in range(num_epochs):
        train_epoch(epoch=epoch,
                    executor=adaptor,
                    x_train=x_train,
                    y_train_onehot=y_train_onehot,
                    batch_size=batch_size,
                    shuffle=True,
                    print_batch_info=False)
        validate_epoch(epoch=epoch,
                       executor=adaptor,
                       x_valid=x_valid,
                       y_valid_onehot=y_valid_onehot,
                       batch_size=batch_size)
        # Export the model configure and snapshot routinely.
        adapt_path = adaptor.export_model_config_with_inputs(inputs=x_train[:batch_size])
        snapshot_path = adaptor.export_model()
        current_costs = adaptor.measure_costs(x_valid)
        satisfied = True
        for i in range(current_costs):
            if current_costs[i] > morphnet_target_costs[i]:
                satisfied = False
                break
        if satisfied:
            print("Satisfied all cost requirements, quit.")
            print("*" * 100)
            break
        if epoch % 10 == 0:
            adaptor.adjust_params(x_train[:batch_size], morphnet_target_costs, initial_costs)
    # End the adaptation
    total_iterations = adaptor.global_step
    snapshot_path = adaptor.export_model()
    adaptor.close()
    return total_iterations, adapt_path, snapshot_path

