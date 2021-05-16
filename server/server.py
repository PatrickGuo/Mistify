"""
Main class for Adaptation coordination
"""

import multiprocessing
import random
import threading
from collections import defaultdict
from multiprocessing import Process

import zmq
import zmq.decorators as zmqd
from termcolor import colored
from zmq.utils import jsonapi

from helper import *
from zmq_decor import multi_sockets
from mistify_worker import AdaptWorker
from utils.graph import ConfigTree, Config


_tf_ver_ = check_tf_version()


class ServerCmd:
    terminate = b'TERMINATION'
    new_job = b'REGISTER'
    fetch_adapted = b'FETCH_ADAPTED_MODELS'
    update_finish = b'UPDATE_FINISH'
    batch_adapt = b'BATCH_ADAPT'
    stream_adapt = b'STREAM_ADAPT'

    @staticmethod
    def is_valid(cmd):
        return any(not k.startswith('__') and v == cmd for k, v in vars(ServerCmd).items())


class MistifyServer(threading.Thread):
    def __init__(self, num_worker=4, port=5555, port_out=5556, cpu=True):
        super().__init__()
        self.logger = set_logger(colored('VENTILATOR', 'magenta'))
        self.num_worker = num_worker
        self.num_concurrent_socket = max(8, num_worker * 2)
        self.port = port
        self.port_out = port_out
        self.cpu = cpu
        self.processes = []
        self.config_tree = ConfigTree()  # conf_id associates to a node in the config tree
        self.finished_models = {}  # conf_id -> path
        self.waiting_configs = {}  # parent_conf_id -> [(childJobId, children_config)]
        self.id2num = {}  # job_id -> jobnum; job_id associates to a specific job (batch or stream)

    def close(self):
        self.logger.info('shutting down...')
        self._send_close_signal()
        for p in self.processes:
            p.close()
        self.join()

    @zmqd.context()
    @zmqd.socket(zmq.PUSH)
    def _send_close_signal(self, _, frontend):
        frontend.connect('tcp://localhost:%d' % self.port)
        frontend.send_multipart([b'', ServerCmd.terminate, b'', b''])

    def run(self):
        self._run()

    @zmqd.context()
    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @multi_sockets(zmq.PUSH, num_socket='num_concurrent_socket')
    def _run(self, _, frontend, sink, *backend_socks):
        # bind all sockets
        self.logger.info('bind all sockets')
        frontend.bind('tcp://*:%d' % self.port)
        addr_front2sink = auto_bind(sink)
        addr_backend_list = [auto_bind(b) for b in backend_socks]
        self.logger.info('open %d frontend-worker sockets' % len(addr_backend_list))

        # start the sink process
        self.logger.info('start the sink')
        proc_sink = MistifySink(addr_front2sink, self.port_out)
        self.processes.append(proc_sink)
        proc_sink.start()
        addr_sink = sink.recv().decode('ascii')

        # start the backend processes
        device_map = self._get_device_map()
        for idx, device_id in enumerate(device_map):
            process = AdaptWorker(idx, addr_backend_list, addr_sink, device_id)
            self.processes.append(process)
            process.start()

        self.rand_backend_socket = None

        def submit_job(parent_path, job_config: Config, job_id, job_num):
            self.rand_backend_socket = random.choice([b for b in backend_socks[1:] if b != self.rand_backend_socket])
            # worker message format: job_id, raw_msg(the adapt_param)
            # Fill in the parent model's base_path
            job_config.full_params['base_path'] = parent_path
            # Notify sink a new job
            sink_msg_dict = {'job_id': job_id, 'job_num': job_num, 'conf_id': job_config.conf_id,
                             'adapt_params': job_config.full_params}
            sink.send_multipart([client, ServerCmd.fetch_adapted, jsonapi.dumps(sink_msg_dict)])
            # Submit the job to worker monitored sockets
            worker_msg_dict = {'conf_id': job_id, 'msg': job_config.full_params}
            worker_msg = jsonapi.dumps(worker_msg_dict)
            self.rand_backend_socket.send_multipart([job_id, job_config.conf_id, worker_msg])

        while True:
            try:
                request = frontend.recv_multipart()
                client, msg_type, msg, req_id, msg_len = request
                job_id = client + b'#' + req_id
            except ValueError:
                self.logger.error('received a wrongly-formatted request')
            else:  # this is reached after running ``try'' statement with no exceptions.
                if msg_type == ServerCmd.terminate:
                    break
                # Submit all ready requests
                # msg fields: path, conf_id
                elif msg_type == ServerCmd.update_finish:
                    message = jsonapi.loads(msg)
                    path = message['adapt_path']
                    snap_path = message['snap_path']
                    conf_id = message['conf_id']
                    self.finished_models[conf_id] = (path, snap_path)
                    for nxtId, config in self.waiting_configs[conf_id]:
                        submit_job(path, config, nxtId, self.id2num[nxtId])
                # Redirect message to sink, client directly ask with a model path.
                # this is called after client receives the model path when finish adaptation
                # sink will reply [(clientaddr), model_data, req_id]
                elif msg_type == ServerCmd.fetch_adapted:
                    self.logger.info('new model fetching request\treq id: %d\tsize: %d\tclient: %s' %
                                     (int(req_id), int(msg_len), client))
                    sink.send_multipart([client, ServerCmd.fetch_adapted, msg])
                # Streaming adaptation
                # JobId -> monitored by client/sink/frontend, it denotes a specific batch/stream job
                # ConfId -> denotes an individual config in the tree, and maps to adapt_path.
                elif msg_type == ServerCmd.stream_adapt:
                    self.logger.info('new streaming adaptation request\treq id: %d\tsize: %d\tclient: %s' %
                                     (int(req_id), int(msg_len), client))
                    message = jsonapi.loads(msg)
                    # Create a new config and insert to config_tree
                    config = Config(self.config_tree.new_id(), message['flops'],
                                    message['ops'], message['adapt_params'])
                    self.config_tree.insert_config(config)
                    adapt_config = self.config_tree.get_adapt_config(config)
                    # Add to waiting config dict if parent is not finished
                    if adapt_config.parent is None:  # no parent, root config
                        submit_job('', adapt_config.current, job_id, 1)
                    elif adapt_config.parent.conf_id not in self.finished_models:
                        if adapt_config.parent.conf_id not in self.waiting_configs:
                            self.waiting_configs[adapt_config.parent.conf_id] = []
                        # adapt configs from different jobs can wait for the same parent config
                        self.waiting_configs[adapt_config.parent.conf_id].append((job_id, adapt_config.current))
                    else:
                        submit_job(self.finished_models[adapt_config.parent.conf_id][1],
                                   adapt_config.current, job_id, 1)
                elif msg_type == ServerCmd.batch_adapt:
                    self.logger.info('new adaptation request\treq id: %d\tsize: %d\tclient: %s' %
                                     (int(req_id), int(msg_len), client))
                    # Compile all the incoming job into config tree
                    adapt_settings = jsonapi.loads(msg)
                    self.id2num[job_id] = len(adapt_settings)
                    self.config_tree = ConfigTree()
                    for setting in adapt_settings:
                        config = Config(self.config_tree.new_id(), setting['flops'],
                                        setting['ops'], setting['adapt_params'])
                        self.config_tree.insert_config(config)
                    # Submit adaptation tasks topologically along the tree
                    # no need to put in waiting because previous ones are always finished
                    adapt_tasks = self.config_tree.topological_sort()
                    for task in adapt_tasks:
                        if task.parent is None:
                            submit_job('', task.current, job_id, self.id2num[job_id])
                        elif task.parent.conf_id not in self.finished_models:
                            if task.parent.conf_id not in self.waiting_configs:
                                self.waiting_configs[task.parent.conf_id] = []
                            self.waiting_configs[task.parent.conf_id].append((job_id, task.current))
                        else:
                            submit_job(self.finished_models[task.parent.conf_id][1],
                                       task.current, job_id, self.id2num[job_id])

        self.logger.info('terminated!')

    def _get_device_map(self):
        self.logger.info('get devices')
        run_on_gpu = False
        device_map = [-1] * self.num_worker
        if not self.cpu:
            try:
                import GPUtil
                num_all_gpu = len(GPUtil.getGPUs())
                avail_gpu = GPUtil.getAvailable(order='memory', limit=min(num_all_gpu, self.num_worker),
                                                maxMemory=0.9, maxLoad=0.9)
                num_avail_gpu = len(avail_gpu)

                if num_avail_gpu >= self.num_worker:
                    run_on_gpu = True
                elif 0 < num_avail_gpu < self.num_worker:
                    run_on_gpu = True
                else:
                    self.logger.warning('no GPU available, fall back to CPU')

                if run_on_gpu:
                    device_map = (avail_gpu * self.num_worker)[: self.num_worker]
            except FileNotFoundError:
                self.logger.warning('nvidia-smi is missing, often means no gpu on this machine. '
                                    'fall back to cpu!')
        self.logger.info('device map: \n\t\t%s' % '\n\t\t'.join(
            'worker %2d -> %s' % (w_id, ('gpu %2d' % g_id) if g_id >= 0 else 'cpu') for w_id, g_id in
            enumerate(device_map)))
        return device_map


class MistifySink(Process):
    def __init__(self, front_sink_addr, port_out=5556):
        super().__init__()
        self.port = port_out
        self.exit_flag = multiprocessing.Event()
        self.logger = set_logger(colored('SINK', 'green'))
        self.front_sink_addr = front_sink_addr

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def run(self):
        self._run()

    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @zmqd.socket(zmq.PUB)
    def _run(self, receiver, frontend, sender):
        receiver_addr = auto_bind(receiver)
        frontend.connect(self.front_sink_addr)
        sender.bind('tcp://*:%d' % self.port)

        # job_id -> AdaptJob(job_id, conf_id, adapt_params)
        pending_jobs = defaultdict()
        # job_id _> num_tasks
        job_nums = defaultdict()
        finished_batch_job_dict = {}

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(receiver, zmq.POLLIN)

        # send worker receiver address back to frontend
        frontend.send(receiver_addr.encode('ascii'))
        logger = set_logger(colored('SINK', 'green'))
        logger.info('ready')

        while not self.exit_flag.is_set():
            socks = dict(poller.poll())
            # Handle worker messages - finished a job
            if socks.get(receiver) == zmq.POLLIN:
                job_id, conf_id, adapt_path, snapshot_path = receiver.recv_multipart()
                client_addr, req_id = job_id.split(b'#')
                # Notify frontend to mark finished configs
                # frontend message format: client, msg_type, msg, req_id, msg_len
                msg_dict = {'adapt_path': adapt_path, 'snap_path': snapshot_path, 'conf_id': conf_id}
                frontend.send_muptipart([client_addr, ServerCmd.update_finish,
                                         jsonapi.dumps(msg_dict), req_id, len(msg_dict)])
                # Update finished job statistics
                job_nums[job_id] -= 1
                finished_batch_job_dict[job_id][conf_id] = (adapt_path, snapshot_path)
                # When all batch jobs finish publish to clients
                if job_nums[job_id] == 0:
                    # publish results to clients
                    sender.send_multipart([client_addr, jsonapi.dumps(finished_batch_job_dict[job_id][0]), req_id])
                    del pending_jobs[job_id]
                    del finished_batch_job_dict[job_id]
                    del job_nums[job_id]

            # Handle frontend messages
            if socks.get(frontend) == zmq.POLLIN:
                client_addr, msg_type, msg_info = frontend.recv_multipart()
                message = jsonapi.loads(msg_info)
                if msg_type == ServerCmd.new_job:
                    # register a new job: two possibilities, 1) single stream job; 2) batch job
                    job_id = message['job_id']
                    if job_id not in job_nums:
                        job_nums[job_id] = int(message['job_num'])
                        pending_jobs[job_id] = {}
                        finished_batch_job_dict[job_id] = {}
                    pending_jobs[job_id][message['conf_id']] = AdaptJob(job_id, message['conf_id'],
                                                                        message['adapt_params'])
                    logger.info('job register\tsize: %d\tjob id: %s' % (int(msg_info), job_id))
                if msg_type == ServerCmd.fetch_adapted:
                    with open(message['path'], 'rb') as f:
                        data = f.read()
                        sender.send_multipart([client_addr, data, message['req_id']])


class AdaptJob:
    def __init__(self, job_id, conf_id, adapt_params: dict):
        self.job_id = job_id
        self.conf_id = conf_id
        self.adapt_params = adapt_params
