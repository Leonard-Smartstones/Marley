# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 08:33:05 2018

@author: Christopher
"""
path = 'C:/Users/Christopher/Marley/BCI/OpenBCI_GUI/OpenBCI_GUI/SavedData/'
fname = 'OpenBCI-RAW-2018-03-27_12-03-21.txt' # 7.5
fname = 'OpenBCI-RAW-2018-03-27_12-04-31.txt' # eyes closed
fname = 'OpenBCI-RAW-2018-03-27_12-20-11.txt' #7.5 again
#fname = 'OpenBCI-RAW-2018-03-27_12-23-45.txt' #12
#fname = 'OpenBCI-RAW-2018-03-27_12-29-34.txt' #12
#fname = 'OpenBCI-RAW-2018-03-27_12-31-31.txt' #10
sample = np.loadtxt(path + fname,
                  delimiter=',',
                  skiprows=7,
                  usecols=(1,2))

sample_ep, _, _ = ama.epoching(sample, 4 * 250, 3.5*250)
sample_psd = ama.rfft_psd(sample_ep[:,0,:], 250)
m = np.mean(sample_psd['PSD'], axis=1)
sample_psd['PSD'] = np.reshape(m, [-1,1])
ama.plot_psd_data(sample_psd, f_range=np.array([0,60]))
'''
for corr_freq in corr_frequencies:
    #plt.figure(figsize=(10,5))
    #ax = plt.subplot(1,1,1)
    sample = getData(HEADSET, filter_, corr_freq=corr_freq)
    
'''