import torch
from torch.distributions import Normal

from dataset import n_node_dataset, WSCRLDataset
from intervset import IntervSet, IntervTable
from plot import plot_dataset

if __name__ == "__main__":
    import networkx as nx


    AUTO = False
    if AUTO:
        # n_node_dataset is a function that genetates MANY graphs
        # of a certain number of nodes based on Ramsay random graphs.
        dataset = n_node_dataset(1, 3, num_samples=50, timesteps=2, markov=2)[0]
    else:
        # FIRST, CREATE A GRAPH
        G = nx.DiGraph()
        # Add edges to the graph
        edges = [('A', 'B'), ('B', 'C'), ('A', 'C')]
        G.add_edges_from(edges)

        # COMPUTE THE SET OF INTERVENTIONS: THE INTERVSET
        x = IntervSet(G, 2)
        print(x.set_of_all_intervs)

        import numpy as np

        # GIVEN THE PRINTED STATEMENT ABOVE, YOU CAN DEFINE YOUR TABLES.
        # (it's also easy to automate this using a forloop on the markov length)
        dict_of_tables = {
            0: np.ones(x.num_interv_ids),
            1: np.random.uniform(0, 10, size=(x.num_interv_ids, x.num_interv_ids)),
            2: np.random.uniform(0, 10, size=(x.num_interv_ids, x.num_interv_ids))
        }
        alpha_vec = np.random.uniform(0.1,1, size=(3,))
        # fixme
        # PASS THE TABLE AND ALPHAS TO THE INTERVSET CALCULATOR
        switch_case = IntervTable(dict_of_tables, alpha_vec)

        x.set_tables(switch_case)
        x.kill(intervs_of_size=2)
        x.kill(intervs_of_size=3)
        temp = x.impossible_intervention_ids

        # DEFINE THE RELATIONSHIP OF EACH NODE TO ITS PARENT
        # (to automate this, just an affine transform given the parents)
        links = {
            'A': lambda parents: Normal(0.0, 1.0).sample(),
            'B': lambda parents: Normal(0.3 * parents[0] ** 2 - 0.6 * parents[0], 0.8 ** 2).sample(),
            'C': lambda parents: Normal(0.2 * parents[0] ** 2 + -0.8 * parents[1], 1.0).sample()
        }

        # DEFINE HOW THE NODES BEHAVE WHEN THEY GET INTERVENED ON
        # (to automate this, just sample from a normal or something of the sort)
        unlinks = {
            'A': lambda: links['A'](None),
            'B': lambda: Normal(0.4, 1.0).sample(),
            'C': lambda: Normal(-0.3, 1.0).sample()
        }

        dataset = WSCRLDataset(1000, 1, G, links, unlinks, intervset=x)

        a = dataset.intervention_ids.unique()

    # To access a single sample
    sample = dataset[0]

    plot_dataset(dataset)
