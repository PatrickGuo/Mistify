"""
Adaptation client
"""

import threading
import time
import uuid
from functools import wraps

import zmq
from server.server import ServerCmd
from zmq.utils import jsonapi


class Client(object):
    def __init__(self, ip='localhost', port=5555, port_out=5556, identity=None, timeout=-1):
        """ A client object connected to a Server
            with Client() as bc:
                bc.encode(...)
        """
        self.context = zmq.Context()
        self.sender = self.context.socket(zmq.PUSH)
        self.sender.setsockopt(zmq.LINGER, 0)  # pending messages discarded.
        self.identity = identity or str(uuid.uuid4()).encode('ascii')
        self.sender.connect('tcp://%s:%d' % (ip, port))

        self.receiver = self.context.socket(zmq.SUB)
        self.receiver.setsockopt(zmq.LINGER, 0)
        self.receiver.setsockopt(zmq.SUBSCRIBE, self.identity)  # filter based on uuid
        self.receiver.connect('tcp://%s:%d' % (ip, port_out))

        self.request_id = 0
        self.timeout = timeout
        self.pending_request = set()
        self.pending_response = {}
        self.port = port
        self.port_out = port_out
        self.ip = ip
        self.length_limit = 0

    def close(self):
        # Close all connections of the client.
        self.sender.close()
        self.receiver.close()
        self.context.term()

    def _send(self, msg_type, msg, msg_len=0):
        self.request_id += 1
        self.sender.send_multipart([self.identity, msg_type, msg, b'%d' % self.request_id, b'%d' % msg_len])
        self.pending_request.add(self.request_id)
        return self.request_id

    def _recv(self, wait_for_req_id=None):
        try:
            while True:
                # a request has been returned and found in pending_response
                if wait_for_req_id in self.pending_response:
                    response = self.pending_response.pop(wait_for_req_id)
                    return wait_for_req_id, response

                # receive a response
                response = self.receiver.recv_multipart()
                request_id = int(response[-1])

                # if not wait for particular response then simply return
                if not wait_for_req_id or (wait_for_req_id == request_id):
                    self.pending_request.remove(request_id)
                    return request_id, response
                elif wait_for_req_id != request_id:
                    self.pending_response[request_id] = response
                    # wait for the next response
        except Exception as e:
            raise e
        finally:
            if wait_for_req_id in self.pending_request:
                self.pending_request.remove(wait_for_req_id)

    def _timeout(self, func):
        """
        Wrapper around client api calls to provide timeout mechanism.
        """
        @wraps(func)
        def arg_wrapper(self, *args, **kwargs):
            if 'blocking' in kwargs and not kwargs['blocking']:
                # override client timeout setting if `func` is called in non-blocking way
                self.receiver.setsockopt(zmq.RCVTIMEO, -1)
            else:
                self.receiver.setsockopt(zmq.RCVTIMEO, self.timeout)
            try:
                return func(self, *args, **kwargs)
            except zmq.error.Again as _e:
                t_e = TimeoutError(
                    'no response from the server (with "timeout"=%d ms), please check the following:'
                    'is the server still online? is the network broken? are "port" and "port_out" correct? '
                    'are you encoding a huge amount of data whereas the timeout is too small for that?' % self.timeout)
                raise t_e from _e
            finally:
                self.receiver.setsockopt(zmq.RCVTIMEO, -1)
        return arg_wrapper

    @_timeout
    def encode(self, adapt_configs, mode):
        """ Encode a list of strings to a list of vectors
            adapt_configs: List of configuration settings.
        """
        if mode == 'stream' and len(adapt_configs) > 1:
            raise ValueError('Stream mode takes one adapt_config at a time.')
        elif mode == 'stream':
            msg_dict = {'flops': adapt_configs[0]['flops'], 'ops': adapt_configs[0]['ops'],
                        'adapt_params': adapt_configs[0]['adapt_params']}
            msg = jsonapi.dumps(msg_dict)
            req_id = self._send(ServerCmd.stream_adapt, msg, len(msg))
        else:
            msg_dict_list = []
            for adapt_config in adapt_configs:
                msg_dict_list.append({'flops': adapt_config['flops'], 'ops': adapt_config['ops'],
                                      'adapt_params': adapt_config['adapt_params']})
            msg = jsonapi.dumps(msg_dict_list)
            req_id = self._send(ServerCmd.batch_adapt, msg, len(msg))
        res_req_id, response = self._recv(req_id)
        return response

    def _fetch(self, delay=.0):
        time.sleep(delay)
        while self.pending_request:
            yield self._recv()

    def fetch_model(self, path, delay=10):
        """ Fetch adapted models
        """
        def run():
            msg_dict = {'path': path}
            self._send(ServerCmd.fetch_adapted, jsonapi.dumps(msg_dict), len(msg_dict))

        t = threading.Thread(target=run)
        t.start()
        res_id, res = self._fetch(delay)
        return res

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

