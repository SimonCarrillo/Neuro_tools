## IsoMap projection
## units in [cells x time], note: isomap algorithm takes the input as time x cells!!
## ndim = dimensions to where you want to project (10 for betti numbers computation) (3 for visualization in 3d plot)
## nneigh = number of neigbors, you will have to systematically increase this parameter and look at the 3d projections and evalute
## wheter they are changing as you increase the number of neigbors, it will depend on the time resolution of your neural activity
## as well as the time scale of the cognitive processes you are interested in observing
def isomap_proj(behavior, units, ndim, nneigh):
    from sklearn.manifold import Isomap
    import numpy as np
    import matplotlib.pyplot as plt    

    tmp_sim = []
    _inp_l = []
    time = []

    _inp_l = units

    Xl = np.sqrt(_inp_l.T)
    print(Xl.shape)

    embedding = []
    embedding = Isomap(n_neighbors=nneigh, n_components=ndim)
    tmp_sim = embedding.fit_transform(Xl)

    time = np.linspace(0, units.shape[1], units.shape[1]) / 5

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    p3 = ax.scatter(tmp_sim[:, 0], tmp_sim[:, 1], tmp_sim[:, 2], c=behavior, s=10, cmap='viridis')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel("Dim 3")
    # ax.axis('equal')
    cb_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(p3, cax=cb_ax, label = 'Dose')
    plt.show()

    behavior = []
    units = []

    return tmp_sim
