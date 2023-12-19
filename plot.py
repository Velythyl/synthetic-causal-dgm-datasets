import matplotlib
import torch

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_tensor(data, dataset):
    IS_3D = data.shape[1] == 3

    interventions = dataset.intervention_ids
    min_interv = interventions[interventions != 0].min()
    max_interv = interventions[interventions != 0].max()
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    cmap = cm.viridis
    norm = Normalize(vmin=min_interv, vmax=max_interv)

    fig = plt.figure(figsize=(12, 12))
    if IS_3D:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()

    color = {
        0: "red",
        1: "green",
        2: "blue",
    }
    for i in range(data.shape[1]):
        ar = data[:, i]
        if IS_3D:
            ax.scatter(ar[:, 0], ar[:, 1], ar[:, 2], color=color[i])
        else:
            ax.scatter(ar[:, 0], ar[:, 1], color=color[i])

    def plot_many_arrows(pairs, color):
        # pairs is of shape [n arrows, 2, (x,y)]

        base_x = pairs[:, 0, 0]
        base_y = pairs[:, 0, 1]
        if IS_3D:
            base_z = pairs[:, 0, 2]

        end_x = pairs[:, 1, 0]
        end_y = pairs[:, 1, 1]
        if IS_3D:
            end_z = pairs[:, 1, 2]

        for i in range(base_x.shape[0]):
            if IS_3D:
                ax.plot([base_x[i], end_x[i]], [base_y[i], end_y[i]], [base_z[i], end_z[i]], color=color)
            else:
                ax.plot([base_x[i], end_x[i]], [base_y[i], end_y[i]], color=color)

    def plot_intervs(data, interventions):
        NUM_INTERVS_OF_EACH_TYPE_TO_PLOT = 2

        for i in interventions.unique():
            if i == 0:
                continue
            # if i not in list(range(dataset.num_nodes+1)): # skips intervs on more than one node
            #    continue

            # opt to select the first elements. doesn't change anything anyway.
            selected_intervs = torch.argsort(interventions == i, descending=True)
            selected_intervs = selected_intervs[:NUM_INTERVS_OF_EACH_TYPE_TO_PLOT].squeeze()
            assert (interventions[selected_intervs] == i).all()

            sel_latents = data[selected_intervs]
            plot_many_arrows(sel_latents, color=cmap(norm(i)))

    plot_intervs(data[:, :2, :], dataset.intervention_ids[:, 0].squeeze())
    if data.shape[0] == 3:
        plot_intervs(data[:, 1:, :], dataset.intervention_ids[:, 1].squeeze())
    plt.show()

def plot_dataset(dataset):
    plot_tensor(dataset.latents, dataset)
    plot_tensor(dataset.observations, dataset)
