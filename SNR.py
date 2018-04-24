# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:54:38 2018

@author: Marley
"""

import numpy as np
import sys
sys.path.append('C://Users//Christopher//Marley//BCI//amplitude-modulation-analysis-module')
from am_analysis import am_analysis as ama
    
class SNR():
    def __init__(self, fs):
        self.headset_freq = fs

    def getResults(self, ch, frequencies):
        x, y = self.get_psd(ch)
        return [self.get_snr(x, y, freq) for freq in frequencies]
    def get_snr(self,freq_ax, psd, freq):
        n = 8
        ix_fs = np.argmin(np.abs(freq_ax - freq))
        sum_background = np.sum(psd[ ix_fs - int(n/2) : ix_fs ]) + np.sum( psd[ ix_fs + 1 : ix_fs + int(n/2) + 1])
        return 10*np.log10( n * psd[ix_fs] / sum_background )
    def get_psd(self, data):
        sample_psd = ama.rfft_psd(data, self.headset_freq)
        return sample_psd['freq_axis'], sample_psd['PSD']
    def getWindowResults(self, sample_ep, frequencies, corr_freq):
        """ Compute the accuracy of SSVEP detection using SNR
            
        Parameters
        ----------
        sample_ep : 
            3D array with shape (n_windows, n_channels, n_samples)
        frequencies : 
            list of frequencies at which we detect SSVEP
        corr_freq : 
            known SSVEP frequency of sample
        
        Returns
        -------
        predictions :
            list of length n_frequencies containing proportion of windows predicted of certain frequency
        accuracy :
            decimal representing accuracy of SSVEP
        all_results:
            2D array with shape (n_windows, n_frequencies) containing SNR or
            3D array with shape (n_windows, n_channels, n_frequencies) containing SNR
        
        """
        right, wrong = 0,0
        all_results = []
        predictions = np.zeros(np.array(frequencies).shape)
        for window in sample_ep:
            results = []
            for ch in window:
                result = self.getResults(ch, frequencies)
                results.append(result)
            
                predictions[np.argmax(result)] += 1
                pred = frequencies[np.argmax(result)]
                #print(pred)
                if (pred == corr_freq):
                    right +=1
                else:
                    wrong +=1
            all_results.append(results)
        all_results = np.squeeze(np.array(all_results))
        print(np.mean(all_results, axis=0))
        print(right+wrong)
        return list(predictions/(right+wrong)), right/(right + wrong), all_results
