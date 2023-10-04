## functions for place cell properties

## useful functions from CircleSquare manuscript
def neighbor_sum(rate_map):
    import numpy as np
    import scipy.ndimage

    mask = np.isnan(rate_map)
    rate_map[mask] = 0

    conv_rate_map = scipy.ndimage.convolve(rate_map, np.ones([3, 3]), mode='constant')

    conv_rate_map[mask] = np.nan
    rate_map[mask] = np.nan
    out_tmp = np.corrcoef(rate_map[~mask].flat, conv_rate_map[~mask].flat)[0, 1]

    return np.mean(out_tmp)


def place_info_content(occ, rate_map):
    import numpy as np
    T = np.sum(occ)  # total time spent
    R = (np.sum(occ[~np.isnan(rate_map)] * rate_map[~np.isnan(rate_map)])) / T  # mean rate
    info_map = (occ / T) * (rate_map / R) * np.log2(rate_map / R)

    return np.sum(info_map[~np.isnan(info_map)])


def nan_gaussian_filter(map, sig):
    import scipy.ndimage
    import numpy as np

    V = map.copy()
    V[np.isnan(map)] = 0
    V_filtered = scipy.ndimage.gaussian_filter(V, sig)

    W = 0 * map.copy() + 1
    W[np.isnan(map)] = 0
    W_filtered = scipy.ndimage.gaussian_filter(W, sig)

    mask = W_filtered < 0.4

    ratio = V_filtered / W_filtered
    out = scipy.ndimage.gaussian_filter(ratio, sig)
    out[mask] = np.nan
    return out


def place_cell_properties(spk_data, beh_data, binsxy, fsrate):
    occ, rate_maps_temp, p_info, coh = [], [], [], []
    nRepeat = 100  # number of shuffles

    # COMPUTING THE RATE MAPS
    occ = np.histogram2d(beh_data[0], beh_data[1], bins=binsxy)[0] / fsrate
    rate_maps_temp = [np.histogram2d(beh_data[0], beh_data[1], bins=binsxy, weights=s)[0]
                      / occ for s in spk_data]

    # analyze rate maps
    p_info = [place_info_content(occ, m) for m in rate_maps_temp]
    coh = [neighbor_sum(m) for m in rate_maps_temp]

    ## pinfo and coherence
    #     print('Place info', np.hstack(p_info))
    #     print('Coherence', np.hstack(coh))

    # Randomizing
    sTrackedCopy = spk_data.copy()
    sTrackedTimePairs = list(zip(*sTrackedCopy))

    pInfoShuffled = []
    cohShuffled = []
    for iRand in range(nRepeat):
        np.random.shuffle(sTrackedTimePairs)
        sShuffled = list(zip(*sTrackedTimePairs))

        RateMapsShuffled = [np.histogram2d(beh_data[0], beh_data[1], bins=binsxy, weights=s)[0]
                            / occ for s in sShuffled]
        pInfoShuffled.append([place_info_content(occ, m) for m in RateMapsShuffled])

        cohShuffled.append([neighbor_sum(m) for m in RateMapsShuffled])

    pInfoTest = []
    for p, z in zip(p_info, zip(*pInfoShuffled)):
        Out = [p, (p - np.mean(z)) / np.std(z)]
        pInfoTest.append([Out, Out[0] > 1 and Out[1] > 1.96])

    cohTest = []
    for p, z in zip(coh, zip(*cohShuffled)):
        Out = [p, (p - np.mean(z)) / np.std(z)]
        testTmp = Out[0] > 0.5 and Out[1] > 1.96
        cohTest.append([Out, testTmp])

    # determine whether cells are place cells
    PC_test = [c[0][1] > 1.96 and p[0][1] > 1.96 for c, p in zip(cohTest, pInfoTest)]
    print('number of PCs', np.sum(PC_test))

    if np.sum(PC_test) > 5:

        ## plot firing rate maps
        for icell in np.random.choice(np.hstack(np.argwhere(np.hstack(PC_test) == True)), 5):
            print('place information', p_info[icell])
            print('coherence', coh[icell])

            fig = plt.figure(figsize=(2, 2))
            ax = fig.add_subplot(1, 1, 1)
            sns.heatmap(rate_maps_temp[icell], cmap='viridis', ax=ax, cbar_kws={'label': 'Normalized rate'})
            ax.set_xlabel('X position')
            ax.set_ylabel('Y position')
            fname = 'Cell' + str(icell)
            ax.set_title(fname)
            plt.show()

    return occ, rate_maps_temp, p_info, coh, PC_test
