import copy
import dataclasses
from itertools import chain, combinations

import networkx as nx
import numpy as np
import torch


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


@dataclasses.dataclass
class Interv:
    self: torch.Tensor
    history: torch.Tensor

    def march_history(self, markov):
        ret = torch.hstack((self.history, self.self[:,None]))
        if ret.shape[1] > markov:
            ret = ret[:,-markov:]
        return ret


def to_tensor(arr):
    if isinstance(arr, torch.Tensor):
        return arr.int()
    assert isinstance(arr, np.ndarray)
    return torch.from_numpy(arr).int()

def to_np(arr):
    if isinstance(arr, np.ndarray):
        return arr.astype(int)
    assert isinstance(arr, torch.Tensor)
    return arr.detach().cpu().numpy().astype(int)

class IntervTable:
    def __init__(self, dict_of_tables, alpha_vec):
        super().__init__()
        self.dict_of_tables = dict_of_tables
        # {
        #   0: (num_intervs,)  # chosen at timestep 0 (ONLY HAPPENS ONCE)
        #   1: (num_intervs, num_intervs) # chosen at order 1
        #   2: (num_intervs, num_intervs) # chosen at order 2
        #   ...
        # }
        self.alpha_vec = alpha_vec
        # {
        #   0: 1    # never used
        #   1: 1    # used, but useless (tragic)
        #   2: (2,) # [0] * dict_of_tables [1] + [1] * dict_of_tables[2]
        #   3: (3,) # [0] * dict_of_tables [1] + [1] * dict_of_tables[2] + [2] * dict_of_tables[3]
        # }
        self.interv_ids = np.arange(dict_of_tables[0].shape[0])

    @classmethod
    def uniform(cls, num_intervs, markov):
        dict_of_tables = {0: np.ones(num_intervs, )}
        dict_of_alphas = {0:1.}
        for m in range(1,markov+1,1):
            dict_of_tables[m] = np.ones((num_intervs, num_intervs))
            dict_of_alphas[m] = 1.

    @property
    def markov(self):
        markov = max(list(self.dict_of_tables.keys()))
        return markov


    @property
    def num_interv_ids(self):
        return self.dict_of_tables[0].shape[0]

    def __call__(self, history):
        if isinstance(history, int) or ( isinstance(history, float) and int(history) == history):
            history = np.array([[]]*history)
        BATCH_SIZE = history.shape[0]
        HISTORY_LENGTH = history.shape[1]

        def maybe_squeeze(arr):
            if len(arr.shape) == 2:
                return arr.squeeze()
            else:
                return arr

        return maybe_squeeze(self.call(BATCH_SIZE, HISTORY_LENGTH, to_np(history)))

    def call(self, BATCH_SIZE, HISTORY_LENGTH, history):
        if HISTORY_LENGTH == 0:
            # edge case: no history to condition on
            weights = self.dict_of_tables[0]
            sum = weights.sum()
            weights = weights / sum
            return np.random.choice(self.interv_ids, size=(BATCH_SIZE,), replace=True, p=weights)

        final_weights = np.zeros((BATCH_SIZE, self.num_interv_ids))
        for i in range(HISTORY_LENGTH):
            weights_for_past = []
            current_alpha = self.alpha_vec[i]
            for l in range(BATCH_SIZE):
                past_node = history[l,i]
                weights_for_past.append(self.dict_of_tables[i+1][past_node] * current_alpha)
            weights_for_past = np.vstack(weights_for_past)
            final_weights += weights_for_past

        sum = final_weights.sum(axis=1)
        sum = np.broadcast_to(sum[:, None], final_weights.shape)
        weights = final_weights / sum

        ret = []
        for i in range(BATCH_SIZE):
            ret.append(np.random.choice(self.interv_ids, size=(1,), replace=True, p=weights[i]))
        ret = np.vstack(ret)
        return ret

class IntervSet:
    def __init__(self, G, markov=0):
        adj_mat = nx.adjacency_matrix(G)
        # Convert the adjacency matrix to a NumPy array (if needed)
        adj_mat = adj_mat.toarray()
        self.markov = markov

        self.num_nodes = adj_mat.shape[0]
        self.adj_mat = adj_mat

        self.set_of_all_intervs = list(sorted(list(powerset(list(range(self.num_nodes))))))
        self.interv_ids = np.arange(len(self.set_of_all_intervs))
        assert self.num_nodes == 1 + max(set([_a for sub in self.set_of_all_intervs for _a in sub]))

        self.probability_tables = None
        self.set_tables(None)

    @property
    def impossible_intervention_ids(self):
        dead = {m: np.zeros(len(self.set_of_all_intervs), dtype=int) for m in range(self.markov+1)}

        dico = copy.deepcopy(self.probability_tables.dict_of_tables)
        for m, prob_table in dico.items():
            for i in self.interv_ids:
                if m == 0:
                    if prob_table[i] == 0:
                        dead[m][i] = 1
                else:
                    if np.all(prob_table[:,i] == 0):
                        dead[m][i] = 1

        always_dead = np.zeros(len(self.set_of_all_intervs), dtype=int)
        for k, v in dead.items():
            always_dead += v
        always_dead = always_dead == self.markov+1

        # dead is a dict that tells you which interv. has P()=0 at each timestep
        # always_dead is a vector that tells you which interv. always has P()=0, no matter the timestep
        return dead, always_dead

    def kill(self, intervs_of_size=None, intervs_in_set=None, nodes_in_intervs=None):
        if intervs_of_size is not None:
            # then it must be
            # 1. a length
            # 2. a dict: (m) -> (length)

            if isinstance(intervs_of_size, int):
                intervs_of_size = {i: intervs_of_size for i in range(self.markov + 1)}

            by_size = np.array([len(_set) for _set in self.set_of_all_intervs])
            intervs_of_size = {m: (by_size == size).nonzero()[0].tolist() for m, size in intervs_of_size.items()}
        else:
            intervs_of_size = {}

        if nodes_in_intervs is not None:
            # 1. a list of intervened nodes. E.g. [(1,2), (), (0,2)]
            # 2. a dict of for timesteps, (m) -> [(1,2),(),(0,2)]
            if isinstance(nodes_in_intervs, dict):
                pass
            else:
                nodes_in_intervs = {i: nodes_in_intervs for i in range(self.markov+1)}

            nodes_in_intervs = {k: [self.set_of_all_intervs.index(tuple(sorted(_v))) for _v in v] for k, v in nodes_in_intervs.items()}
        else:
            nodes_in_intervs = {}

        if intervs_in_set is not None:
            # if it is not None, then it must be:
            # 1. a set of nodes
            # 2. a dict: (m) -> (set of nodes)

            if isinstance(intervs_in_set, dict):
                pass
            else:
                intervs_in_set = {i: intervs_in_set for i in range(self.markov+1)}
        elif intervs_in_set is None:
            intervs_in_set = {}

        def merge_dict(d1, d2):
            ret = {}
            done_keys = set()
            for k, v in d1.items():
                if k in d2:
                    ret[k] = v + d2[k]
                else:
                    ret[k] = v
                done_keys.add(k)
            for k, v in d2.items():
                if k in done_keys:
                    continue
                ret[k] = v
            return ret

        intervs_in_set = merge_dict(intervs_in_set, intervs_of_size)
        intervs_in_set = merge_dict(intervs_in_set, nodes_in_intervs)

        dico = copy.deepcopy(self.probability_tables.dict_of_tables)
        for m, list_of_nodes in intervs_in_set.items():
            for n in list_of_nodes:
                if m == 0:
                    dico[m][n] = 0.0
                else:
                    dico[m][:,n] = 0.0

        self.probability_tables.dict_of_tables = dico

    def id2interv(self, id):
        return self.set_of_all_intervs[id]

    def n_m_onehots(self, batch_ids):
        # (batch size, markov,)

        # this messed up looking 3-for-loop piece of shit just builds one-hots
        interventions = []
        for s in range(batch_ids.shape[0]):
            ret = []
            for m in range(batch_ids.shape[1]):
                vec = torch.zeros(self.num_nodes).int()
                set_of_intervened_nodes = self.id2interv(batch_ids[s,m])
                for n in set_of_intervened_nodes:
                    vec[n] = 1
                ret.append(vec)
            interventions.append(torch.stack(ret))
        interventions = torch.stack(interventions)
        return interventions

    def onehots_to_tuples(self, batch_onehots):
        interventions = []
        for s in range(batch_onehots.shape[0]):
            sub = []
            for m in range(batch_onehots.shape[1]):
                sub.append(tuple(batch_onehots[s,m].nonzero().unique().cpu().numpy()))
            interventions.append(sub)
        return interventions

    @property
    def num_interv_ids(self):
        return self.interv_ids.shape[0]

    def set_tables(self, switch_case):
        if switch_case is None:
            # default to uniform
            switch_case = IntervTable.uniform(self.num_interv_ids, self.markov)
        self.probability_tables = switch_case

    def init(self, batch_size):
        if self.probability_tables is None:
            raise Exception("how dare you")

        return Interv(to_tensor(self.probability_tables(batch_size)), to_tensor(np.array([[]] * batch_size)))

    def pick(self, past_interv):
        if self.probability_tables is None:
            raise Exception("how dare you")

        history = past_interv.march_history(self.markov)

        return Interv(to_tensor(self.probability_tables(history)), history)
