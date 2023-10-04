## compute persistent homology using betti numbers
## input is an ISOMAP projection of your spiking activity to a 10-Dimensional ISOmap space
def compute_betti(dataIso):
    from scipy.spatial.distance import pdist
    from sklearn import neighbors
    from ripser import ripser as tda
    from sklearn.manifold import Isomap
    import numpy as np
    import matplotlib.pyplot as plt        

    dataBetti = []
    H1_rates = []
    H2_rates = []
    barcodes = []
    dist = []
    rad = []
    neigh = []
    num_nbrs = []
    threshold = []
    thrsh_rates = []
    results = []
    idx = []

    Bettithrsh = False  # treshold nb of data point to h2MaxDt
    doBetti = True
    h2MaxDt = 700

    if Bettithrsh:
        #         print('neighbor thresholding 2')
        # a) find number of neighbors of each point within radius of 1st percentile of all
        # pairwise dist.
        dist = pdist(dataIso, 'euclidean')
        rad = np.percentile(dist, 1)
        neigh = neighbors.NearestNeighbors()
        neigh.fit(dataIso)
        num_nbrs = [*map(len, neigh.radius_neighbors(X=dataIso, radius=rad, return_distance=False))]

        # b) threshold out points with low density
        thrsh_prcnt = 20
        threshold = np.percentile(num_nbrs, thrsh_prcnt)
        thrsh_rates = dataIso[num_nbrs > threshold]
        dataBetti = thrsh_rates
    else:
        dataBetti = dataIso

    #     print('computing Betti numbers...')
    results = {'h0': [], 'h1': [], 'h2': []}
    # Betti
    # H0 & H1
    H1_rates = dataBetti
    #     print('h0-1')
    barcodes = tda(H1_rates, maxdim=1, coeff=2)['dgms']
    results['h0'] = barcodes[0]
    results['h1'] = barcodes[1]
    if len(dataBetti) > h2MaxDt:
        #         print('shortening data for Betti nb 2')
        idx = np.random.choice(np.arange(len(dataBetti)), h2MaxDt, replace=False)
        H2_rates = dataBetti[idx]
    else:
        H2_rates = dataBetti
    #     print('h2')
    barcodes = tda(H2_rates, maxdim=2, coeff=2)['dgms']
    results['h2'] = barcodes[2]
    #     print('done')

    plot_barcode = True

    h0, h1, h2 = results['h0'], results['h1'], results['h2']
    # replace the infinity bar (-1) in H0 by a really large number (this will depend on the spread of your data in isomap space)
    h0[~np.isfinite(h0)] = 50
    h1[~np.isfinite(h1)] = 50
    h2[~np.isfinite(h2)] = 50
    # Plot the longest barcodes only
    plot_prcnt = [99, 98, 90]  # order is h0, h1, h2

    if plot_barcode:
        print('plotting...')
        col_list = ['r', 'g', 'm', 'c']
        to_plot = []
        for curr_h, cutoff in zip([h0, h1, h2], plot_prcnt):
            bar_lens = curr_h[:, 1] - curr_h[:, 0]
            if len(curr_h) > 0:
                plot_h = curr_h[bar_lens > np.percentile(bar_lens, cutoff)]
                to_plot.append(plot_h)

        for curr_betti, curr_bar in enumerate(to_plot):
            #             ax = fig.add_subplot()
            figBetti = plt.figure(figsize=(15, 15))
            ax = figBetti.add_subplot(3, 9, curr_betti * 9 + 1)
            for i, interval in enumerate(reversed(curr_bar)):
                ax.plot([interval[0], interval[1]], [i, i], '.-', color=col_list[curr_betti],
                        lw=1.5)
            # ax.set_xlim([0, xlim])
            # ax.set_xticks([0, xlim])
            ax.set_ylim([-1, 55])  # len(curr_bar)])
            ax.set_xlim([-1, 55])
            # ax.set_yticks([])
        plt.show()
        print('plotted')

    data_betti = h0, h1, h2

    betti_data = []
    h1_ratio = []
    h1_max = []
    h1 = []

    return data_betti
