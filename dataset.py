import dataclasses

import networkx as nx
import numpy as np
import torch
from networkx import NetworkXNoCycle
from torch.distributions import Normal
from torch.utils.data import Dataset

from encoder.flow import FlowEncoder
from encoder.transforms import make_scalar_transform
from generate_for_graph import generate, node_to_index, roots
from intervset import IntervSet, IntervTable


def maybe_detach(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach()
    return arr


@dataclasses.dataclass
class WSCRLData:
    latents: torch.Tensor
    observations: torch.Tensor


class WSCRLDataset(Dataset):
    def __init__(self, num_samples, timesteps, G, links, unlinks, intervset, timestep_carryover=True):
        self.num_samples = num_samples
        self.intervset = intervset

        self.latents, self.observations, self.interventions, self.intervention_ids = generate(num_samples, timesteps, G,
                                                                                              links, unlinks, intervset,
                                                                                              timestep_carryover=timestep_carryover)

        self.latents = maybe_detach(self.latents)
        self.observations = maybe_detach(self.observations)
        self.interventions = maybe_detach(self.interventions)
        self.intervention_ids = maybe_detach(self.intervention_ids)

    @property
    def markov(self):
        return self.intervset.markov

    @property
    def num_interv_types(self):
        # fixme big assumption: all interventions were sampled during generation (this might not be true)
        return self.intervention_ids.unique().shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.observations[idx], self.latents[idx], self.intervention_ids[idx], self.interventions[idx]


class AutomaticDataset(WSCRLDataset):
    def __init__(self, num_samples, timesteps, markov, G, timestep_carryover):
        try:
            nx.find_cycle(G)
            raise AssertionError("Graph can't have cycles")
        except NetworkXNoCycle:
            pass

        n2i = node_to_index(G)
        starts = roots(G)
        descendents = set(list(G.nodes)) - starts
        assert len(starts) != 0

        links = {k: lambda parents: Normal(n2i[k], 1.0).sample() for k in starts}
        unlinks = {k: lambda: v(None) for k, v in links.items()}

        for node in descendents:
            # find number of parents
            n_parents = len(list(G.predecessors(node)))

            def make_link():
                #flow_encoder = FlowEncoder(
                #    input_features=n_parents,
                #    output_features=n_parents,
                #    transform_blocks=2  # todo maybe add blocks
                #)
                flow_encoder = make_scalar_transform(n_parents, layers=3)

                def descendant_link(parents):
                    flow = flow_encoder(parents[None])[0]

                    if n2i[node] % 2 == 0:
                        flow = flow.mean()
                    else:
                        flow = flow.sum()

                    return Normal(0.0, 1.0).sample() + flow

                return descendant_link

            links[node] = make_link()
            unlinks[node] = lambda: Normal(0.1 * n2i[node], 1.0).sample()

        intervset = IntervSet(G, markov)

        def random_uniform(is_vec):
            return np.random.uniform(0, 10, size=(intervset.num_interv_ids,) if is_vec else (
            intervset.num_interv_ids, intervset.num_interv_ids))

        dict_of_tables = {i: random_uniform(False) for i in range(markov + 1)}
        dict_of_tables[0] = random_uniform(True)

        alpha_vec = np.random.uniform(0.1, 1, size=(markov+1))

        # PASS THE TABLE AND ALPHAS TO THE INTERVSET CALCULATOR
        switch_case = IntervTable(dict_of_tables, alpha_vec)
        intervset.set_tables(switch_case)

        super().__init__(num_samples, timesteps, G, links, unlinks, intervset, timestep_carryover)


def n_node_dataset(num_datasets, num_nodes_OR_generator, num_samples, timesteps, markov):
    if isinstance(num_nodes_OR_generator, int):
        def has_cycle(g):
            try:
                nx.find_cycle(g)
                return True
            except:
                return False

        def gen_graph(i):
            def gen():
                return nx.fast_gnp_random_graph(num_nodes_OR_generator, 0.7, directed=True)

            g = gen()
            while has_cycle(g):
                g = gen()

            return g

        generator = gen_graph
    else:
        generator = num_nodes_OR_generator

    ret = []
    while len(ret) != num_datasets:
        graph = generator(len(ret))
        ret += [AutomaticDataset(num_samples, timesteps, markov, graph, timestep_carryover=False)]
    return ret


