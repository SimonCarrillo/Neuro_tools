## Kendall Tau correlations parallized version
## you can also use pearson or any kind of pairwise computation
### features [cells x time]
def kendall_tau_parallel(features):
    import itertools
    import numpy as np
    import scipy.stats as ss
    from tqdm.notebook import tqdm
    import dask
    from dask.diagnostics import ProgressBar

    ## all cells
    tau_corrs_list = []
    tauVec = []
    corr_mat = []
    units_common = []

    units_common = np.arange(0, features.shape[0], 1)
    #     print(units_common)

    # keep pairs id
    cell_pairs = []
    cell_pairs = [(x1, x2) for x1, x2 in itertools.combinations(units_common, 2)]

    print(len(cell_pairs))

    # delayed_results = [dask.delayed(ss.pearsonr)(features[icell1,:],features[icell2,:],) for cell1, cell2 in tqdm(cell_pairs)]
    delayed_results = [dask.delayed(ss.kendalltau)(features[cell1, :], features[cell2, :], ) for cell1, cell2 in
                       tqdm(cell_pairs)]

    with ProgressBar():
        tau_corrs_list = dask.compute(delayed_results, scheduler='processes')
    tauVec = np.array(tau_corrs_list).copy()
    tau_corrs_list = [corr[0] for corr in tau_corrs_list[0]]  # remove p-values from list
    # put corrs list into matrix
    corr_mat = np.zeros((features.shape[0], features.shape[0]))

    for i, (cell1, cell2) in enumerate(cell_pairs):
        corr_mat[cell1, cell2] = tau_corrs_list[i]
        corr_mat[cell2, cell1] = tau_corrs_list[i]
    np.fill_diagonal(corr_mat, 1)

    return corr_mat, cell_pairs, tauVec
