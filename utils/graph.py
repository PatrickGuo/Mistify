"""
Topological sorting utilities & configuration tree holder
"""
from collections import defaultdict


class Config:
    def __init__(self, conf_id, flops, ops, full_params):
        self.conf_id = conf_id  # uid of this adaptation goal
        self.flops = flops
        self.ops = ops
        self.full_params = full_params

    def is_stricter(self, other):
        return self.flops < other.flops and self.ops < other.ops


class AdaptConfig:
    def __init__(self, parent: Config, current: Config):
        self.parent = parent
        self.current = current


# Class to represent a graph
class ConfigTree:
    def __init__(self):
        self.graph = defaultdict(list)  # dictionary containing adjacency List
        self.parents = defaultdict(list)  # forward dependency
        self.configs = []  # all the configs
        self.conf2idx = {}

    def new_id(self):
        return len(self.configs)

    def insert_config(self, conf):
        id = len(self.configs)
        self.configs.append(conf)
        self.conf2idx[conf] = id
        for idx in range(len(self.configs)):
            if idx == id:
                continue
            if self.configs[id].compare(self.configs[idx]) < 0:  # new config is stricter
                self.add_edge(idx, id)
                self.parents[id].append(idx)

    # function to add an edge to graph
    def add_edge(self, u, v):
        self.graph[u].append(v)

    # A recursive function used by topologicalSort
    def topological_sort_util(self, v, visited, stack):
        visited[v] = True
        for i in self.graph[v]:
            if not visited[i]:
                self.topological_sort_util(i, visited, stack)
        par = None
        for p in self.parents[v]:
            nxtpar = self.configs[p]
            if par is None or nxtpar.is_stricter(par):
                par = nxtpar
        stack.insert(0, AdaptConfig(par, v))

    # The function to do Topological Sort. It uses recursive
    # topologicalSortUtil()
    def topological_sort(self):
        visited = [False] * len(self.configs)
        stack = []
        for i in range(len(self.configs)):
            if not visited[i]:
                self.topological_sort_util(i, visited, stack)
        return stack

    # For stream adaptation usage
    def get_adapt_config(self, config):
        idx = self.conf2idx[config]
        par = None
        for pidx in self.parents[idx]:
            nxtpar = self.configs[pidx]
            if par is None or nxtpar.is_stricter(par):
                par = nxtpar
        return AdaptConfig(par, config)
