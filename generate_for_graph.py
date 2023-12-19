import dataclasses
import functools

import torch
from tqdm import tqdm

import networkx as nx

from encoder.flow import FlowEncoder
from encoder.transforms import make_scalar_transform


@functools.lru_cache
def node_to_index(G):
    nodes = list(G.nodes())
    node_to_index = {node: index for index, node in enumerate(nodes)}
    return node_to_index


@functools.lru_cache
def index_to_node(G):
    nodes = list(G.nodes())
    index_to_node = {index: node for index, node in enumerate(nodes)}
    return index_to_node


@functools.lru_cache
def reachables(G, source_ids):
    source_names = map(lambda i: index_to_node(G)[i], source_ids)

    reachables = set()
    for node in source_names:
        reachable_from_node = set(nx.descendants(G, node))

        reachables.update(reachable_from_node)
        reachables.add(node)

    return list(map(lambda i: node_to_index(G)[i], reachables))


@functools.lru_cache
def parents_dict(G):
    n2i = node_to_index(G)

    ret = {}
    for n, i in n2i.items():
        ret[i] = torch.tensor(list(sorted([n2i[parent] for parent in G.predecessors(n)]))).int()
    return ret


@functools.lru_cache
def execution_order(G):
    return [node_to_index(G)[n] for n in nx.topological_sort(G)]


@functools.lru_cache
def roots(G):
    ret = []
    for n in G.nodes:
        if len(G.predecessors(n)) == 0:
            ret.append(n)
    return ret


@functools.lru_cache
def intervened_execution_order(G, intervened_nodes):
    ret = []
    reachee = reachables(G, intervened_nodes)
    for i in execution_order(G):
        if i in reachee:
            ret += [i]
    return ret


@functools.lru_cache
def num_nodes(G):
    return len(G.nodes())


@functools.lru_cache
def roots(G):
    starts = set([node for node in G.nodes if G.in_degree(node) == 0])
    return starts


def generate_one(interventions, G, links, unlinks, timestep_carryover):
    def sample_node(node_id, parent_data):
        link = links[node_id]
        ret = link(parent_data)
        return ret

    def sample_node_interv(node_id):
        return unlinks[node_id]()

    vec = torch.zeros(num_nodes(G))
    for node in execution_order(G):
        parents = parents_dict(G)[node]
        if len(parents) >= 0:
            parent_data = vec[parents]
        else:
            parent_data = None
        vec[node] = sample_node(node, parent_data)

    data = [vec]

    for m in range(1, len(interventions) + 1, 1):
        data += [torch.clone(data[m - 1])]

        set_of_intervened_nodes = interventions[m - 1]
        if len(set_of_intervened_nodes) == 0:
            continue

        if timestep_carryover:
            # if there IS copyover, then we skip the nodes that are unreached by the interventions. We only resample the intervened nodes and their downstream
            nodelist = intervened_execution_order(G, set_of_intervened_nodes)
        else:
            # if there IS NOT copyover, then we resample everything
            nodelist = execution_order(G)

        for node in nodelist:
            if node in set_of_intervened_nodes:
                val = sample_node_interv(node)
            else:
                val = sample_node(node, data[-1][parents_dict(G)[node]])

            data[m][node] = val
    return data


def generate(num_samples, timesteps, G, links, unlinks, intervset, timestep_carryover):
    links = {node_to_index(G)[k]: v for k, v in links.items()}
    unlinks = {node_to_index(G)[k]: v for k, v in unlinks.items()}
    num_nodes = len(G.nodes())

    flow_encoder = make_scalar_transform(num_nodes, layers=3)
    #flow_encoder = FlowEncoder(
    #    input_features=num_nodes,
    #    output_features=num_nodes,
    #    transform_blocks=1,
    #    layers=1,
    #    sigmoid=True
    #)

    ALL_INTERVENTIONS = [
        intervset.init(num_samples)]  # 1 is batch_size (here we are generating point-by-point, so it's 1
    for m in range(timesteps - 1):
        ALL_INTERVENTIONS += [intervset.pick(ALL_INTERVENTIONS[-1])]
    intervention_ids = torch.stack([i.self for i in ALL_INTERVENTIONS]).T
    interventions = intervset.n_m_onehots(intervention_ids)

    latents = []
    observations = []
    intervention_tuples = intervset.onehots_to_tuples(interventions)

    for i in tqdm(range(num_samples)):
        lat = generate_one(intervention_tuples[i], G, links, unlinks, timestep_carryover)
        lat = torch.stack(lat)
        observations.append(flow_encoder(lat)[0])
        latents.append(lat)

    latents = torch.stack(latents)
    observations = torch.stack(observations)

    return latents, observations, interventions, intervention_ids
