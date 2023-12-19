# synthetic-causal-dgm-datasets
Obtain samples for any valid causal DGM. Explicitly define the graph &amp; its interventions or automatically generate them both!

# Explicit graph, dependencies, and interventions

You can sample data from explicit graphs.

Create the graph

```python
# FIRST, CREATE A GRAPH
G = nx.DiGraph()
# Add edges to the graph
edges = [('A', 'B'), ('B', 'C'), ('A', 'C')]
G.add_edges_from(edges)
```

Given that graph, we can compute the set of all possible interventions. This is the powerset of the nodes.
```python
x = IntervSet(G, 2)
print(x.set_of_all_intervs) # [(), (0,), (0, 1), (0, 1, 2), (0, 2), (1,), (1, 2), (2,)]
```

Given that powerset, you can manually define your tables
```python
dict_of_tables = {
    0: np.ones(x.num_interv_ids),   # first interv. doesn't depend on past
    1: np.random.uniform(0, 10, size=(x.num_interv_ids, x.num_interv_ids)),  # all subsequent interventions depend on past
    2: np.random.uniform(0, 10, size=(x.num_interv_ids, x.num_interv_ids))
}
alpha_vec = np.random.uniform(0.1,1, size=(3,)) # defines the contributing weight of each table in the dict_of_tables
# fixme
# PASS THE TABLE AND ALPHAS TO THE INTERVSET CALCULATOR
switch_case = IntervTable(dict_of_tables, alpha_vec)
x.set_tables(switch_case)

# You can also ``kill'' interventions, i.e. set their probability to 0, 
# irrespective of your tables defined above. This is useful when you
# care about preventing some interventions.
x.kill(intervs_of_size=2)
x.kill(intervs_of_size=3)

# In any case, you can get the interventions with P=0 for specific timesteps
# get looking at this property
temp = x.impossible_intervention_ids
```

Now that we have out graph and our set of interventions, we need to define the relationship between the nodes ("links")
and the effects of the interventions ("unlink").

```python
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
```

Finally, pass all of this to the dataset generator.
```python
dataset = WSCRLDataset(1000, 1, G, links, unlinks, intervset=x)
```

As a reminder, this "explicit" way of doing this, while tedious, allows you to simulate specific graphs.

# Automatic 

You can forgo all the tediousness above by generating random graphs, random dependencies, random intervention probability tables, random (...).
You get it.

Just call this:
```python
dataset = n_node_dataset(num_datasets=1, num_nodes=3, num_samples=50, timesteps=2, markov=2)[0]
```

This generates <num_datasets> random connected graphs of <num_nodes> nodes, get <num_samples> of <timesteps+1> number of points each.