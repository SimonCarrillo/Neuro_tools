## spk is cell x time
## x and y positions in time
## win_legth which is the length of the path

def overdispersion(spk,xframes,yframes,iPosframes,win_length, thresh, frame, ipos_mode):

    import numpy as np
    import pandas as pd

    
    Nbins = 16
    edgelength = 256/Nbins
    edges = np.arange(0,257,edgelength)
    sampRate = 7.5 # number of samples in a second

    occ_all = []
    rate_maps_all = []
    mean_fr_all = []
    mask_high_fr_all = []
    mask_deltaipos_all = []
    observed_rate_all = []
    expected_rate_all = []
    z_all = []
    
    binx_all = []
    biny_all = []
        
    if frame == "room":
        x,y,ipos = [],[],[]
        x = np.squeeze(xframes[0])
        y = np.squeeze(yframes[0])  
        ipos = iPosframes[0][0] 
        
    if frame == "arena":
        x,y,ipos = [],[],[]
        x = np.squeeze(xframes[1])
        y = np.squeeze(yframes[1])   
        ipos = iPosframes[0][1]      
    print('ipos_shape',ipos.shape)
    
    deltaipos = []
    deltaipos = iPosframes[0][0] - iPosframes[0][1] 
    
    # COMPUTING THE RATE MAPS
    for icell in range(0,spk.shape[0]):
        occ = []
        rate_maps_temp = []

        ## remove nans

        idx_nan = []
        idx_nan = np.invert(np.isnan(x) | np.isnan(y))
        
        mean_fr = []
        mean_fr = np.nansum(spk[icell])/(spk[icell].size/sampRate)
        

        # compute position map
        occ = np.histogram2d(x[idx_nan], y[idx_nan], bins=edges)[0]/sampRate
        # compute firing rate map normalized by occupancy
        rate_maps_temp = [np.histogram2d(x[idx_nan], y[idx_nan], bins=edges, weights=spk[icell][idx_nan])[0] / occ]

        # plot firing rate map per cell
        figure, axes = plt.subplots(figsize=(3,3))         
        sns.heatmap(rate_maps_temp[0])
        plt.show()
        
        
        ## not sure if that is real
        if mean_fr > 2:
#         if np.max(rate_maps_temp[0]) > 20 or np.max(rate_maps_temp[0]) < 10:
            print('high mean fr', mean_fr)
            print('PC peak rate',np.max(rate_maps_temp[0]))
            continue
        
        mean_fr_all.append(mean_fr)
        occ_all.append(occ)
        rate_maps_all.append(rate_maps_temp)
        
        ## compute overdispersion in windows of 5 seconds
        
        n_windows = []
        n_windows = spk[icell][idx_nan].size // win_length
        print('total number of epochs', n_windows)
        print('length of spiking in samples',spk[icell][idx_nan].size)
        spk_split = []
        
        if (spk[icell][idx_nan].size % win_length) > 0:
            spk_split = np.split(spk[icell][idx_nan][:-(spk[icell][idx_nan].size % win_length)],n_windows)
        if (spk[icell][idx_nan].size % win_length) == 0:
            spk_split = np.split(spk[icell][idx_nan],n_windows)
            
        print('spk.length',spk[icell][idx_nan].size)
        print('one path size',spk_split[0].shape)
        print('check')
        print(spk_split[0])
        print(spk[icell][idx_nan][0:30])

        if (spk[icell][idx_nan].size % win_length) > 0:

            x_split = []
            x_split = np.split(x[idx_nan][:-(spk[icell][idx_nan].size % win_length)],n_windows)
            y_split = []
            y_split = np.split(y[idx_nan][:-(spk[icell][idx_nan].size % win_length)],n_windows)   

            ## per time point compute expected rate, firing rate map sum over time

            bin_x = []
            bin_y = []       

            bin_x = [np.max([0,np.where(x_curr <= edges)[0][0]-1]) for x_curr in x[idx_nan][:-(spk[icell][idx_nan].size % win_length)]]
            bin_y = [np.max([0,np.where(y_curr <= edges)[0][0]-1]) for y_curr in y[idx_nan][:-(spk[icell][idx_nan].size % win_length)]]

        if (spk[icell][idx_nan].size % win_length) == 0:

            x_split = []
            x_split = np.split(x[idx_nan],n_windows)
            y_split = []
            y_split = np.split(y[idx_nan],n_windows)   

            ## per time point compute expected rate, firing rate map sum over time

            bin_x = []
            bin_y = []       

            bin_x = [np.max([0,np.where(x_curr <= edges)[0][0]-1]) for x_curr in x[idx_nan]]
            bin_y = [np.max([0,np.where(y_curr <= edges)[0][0]-1]) for y_curr in y[idx_nan]]
            
        ### ipos per cell
        if ipos_mode == "cell":
            if (spk[icell][idx_nan].size % win_length) > 0:
                deltaipos_split = np.split(deltaipos[icell][idx_nan][:-(spk[icell][idx_nan].size % win_length)],n_windows)
            if (spk[icell][idx_nan].size % win_length) == 0:
                deltaipos_split = np.split(deltaipos[icell][idx_nan],n_windows)        
 
        ### ipos as a population
        if ipos_mode == "population":
            if (spk[icell][idx_nan].size % win_length) > 0:
                deltaipos_split = np.split(np.nansum(deltaipos,axis=0)[idx_nan][:-(spk[icell][idx_nan].size % win_length)],n_windows)
            if (spk[icell][idx_nan].size % win_length) == 0:
                deltaipos_split = np.split(np.nansum(deltaipos,axis=0)[idx_nan],n_windows)        
        
        expected_timeseries = []
        expected_timeseries = [rate_maps_temp[0][bx,by]/sampRate  for bx,by in zip(bin_x,bin_y)]
#         expected_timeseries = [rate_maps_temp[0][bx,by]/sampRate  for bx,by in zip(bin_x,bin_y)]

        expected_timeseries_split = []        
        expected_timeseries_split = np.split(np.concatenate(expected_timeseries, axis=None),n_windows)   

        observed_rate, expected_rate = [], []

        # compute observed rate in the epoch
        observed_rate = [np.sum(p) for p in spk_split]

        #compute expected rate in the epoch from mean firing rate map
        expected_rate = [np.sum(p) for p in expected_timeseries_split]

        mask_highfr = []
        mask_highfr = [s > mean_fr*thresh for s in expected_rate]
        
        mask_ipos = []
        mask_ipos = [np.nanmean(s)>0 for s in deltaipos_split]
        
        mask_deltaipos_all.append(mask_ipos) ## positive means room frame

        z = []
        z = [(o-e)/np.sqrt(e) for o,e in zip(observed_rate,expected_rate)]

        mask_high_fr_all.append(mask_highfr)
        observed_rate_all.append(observed_rate)
        expected_rate_all.append(expected_rate)
        z_all.append(z)
        
        binx_all.append(bin_x)
        biny_all.append(bin_y)
        
        ## plot expected and observed rate
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1)
        win = np.arange(0,n_windows,1)
        ax.scatter(win, np.hstack(observed_rate)/(win_length/sampRate), color = 'black')
        ax.scatter(win, np.hstack(expected_rate)/(win_length/sampRate), color = 'red')
        ax.plot(mean_fr*np.ones(300), '--', color = 'gray')

        ax.set_ylabel('spikes per second')
        ax.set_xlabel('Path #')
        plt.show()
        
        ## plot z
        
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(1,1,1)
        win = np.arange(0,n_windows,1)
        ax.scatter(win, z, color = 'black')
        ax.set_ylabel('Z')
        ax.set_xlabel('Path #')
        plt.show()        
        
    x, y, spk = [], [], []
        
    return occ_all, rate_maps_all, mean_fr_all, mask_high_fr_all,mask_deltaipos_all, observed_rate_all,expected_rate_all,z_all, binx_all, biny_all
