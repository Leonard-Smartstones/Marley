# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 10:05:36 2018

@author: Marley
"""
import numpy as np

def getWindowSNR(sample_ep, frequencies, corr_freq, window_length, template=None):
    
    right, wrong = 0,0
    all_results = []
    predictions = np.zeros(np.array(frequencies).shape)
    for window in sample_ep:
        for ch in window:
            x, y = get_psd(ch)
            results = np.array([get_snr(x, y, freq) for freq in frequencies])
            all_results.append(results)
        
            predictions[np.argmax(results)] += 1
            pred = frequencies[np.argmax(results)]
            print(pred)
            #print(pred)
            if (pred == corr_freq):
                right +=1
            else:
                wrong +=1
    all_results = np.array(all_results)
    print(np.mean(all_results, axis=0))
    print(right+wrong)
    return list(predictions/(right+wrong)), right/(right + wrong), all_results
def get_snr(freq_ax, psd, freq):
    n = 8
    ix_fs = np.argmin(np.abs(freq_ax - freq))
    sum_background = np.sum(psd[ ix_fs - int(n/2) : ix_fs - 1 ]) + np.sum( psd[ ix_fs + 1 : ix_fs + int(n/2) ])
    return 10*np.log10( n * psd[ix_fs] / sum_background )
def get_psd(data):
    sample_psd = ama.rfft_psd(data, HEADSET_FREQ)
    return sample_psd['freq_axis'], sample_psd['PSD']
frequencies = [7.5, 10, 12]
corr_freq = 7.5
sample = getData(HEADSET, filter_, corr_freq=corr_freq)
sample_ep = epoch_data(sample[[0]], window_length, pred_interval)
all_acc, acc, all_resu = getWindowSNR(sample_ep, frequencies, corr_freq, window_length)

idx = frequencies.index(corr_freq)
pred = np.argmax(np.squeeze(all_resu), axis = -1)
num_corr = 0
num = 0
arr = []
for i in pred:
    if i == idx:
        num_corr +=1
    num += 1
    arr.append(num_corr/num)
    
frequencies = [7.5, 10, 12, 'Baseline']
for f in frequencies:
    sample = getData(HEADSET, filter_, corr_freq=f)
    plot_average_spectrum(sample[[0]], window_length, pred_interval)