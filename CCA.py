# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:24:11 2018

@author: Marley
"""
import numpy as np
from sklearn.cross_decomposition import CCA
    
class CcaExtraction():
    def __init__(self, HEADSET_FREQ):
        self.cca = CCA(n_components=1)
        self.headset_freq = HEADSET_FREQ
        self.harmonic_coeff = [1,2,3]
        self.reference_signals = None
        
    def getReferenceSignals(self, length, target_freqs):
        """ Generate and set the pure frequency templates
            
        Parameters
        ----------
        length : 
            length of template in samples
        target_freqs : 
            list of frequencies at which we detect SSVEP
        
        Returns
        -------
        self.reference_signals : 
            3D array of shape (n_frequencies, 6, length)
        
        """   
        reference_signals = []
        t = np.arange(0, length, step=1.0)/self.headset_freq
        self.harmonics = [self.harmonic_coeff]*len(target_freqs)
        for freq, harmonics in zip(target_freqs, self.harmonics):
            reference_signals.append([])
            for harmonic in harmonics:
                reference_signals[-1].append(np.sin(np.pi*2*harmonic*freq*t))
                reference_signals[-1].append(np.cos(np.pi*2*harmonic*freq*t))
        self.reference_signals = np.array(reference_signals)
        return self.reference_signals

    def getCorr(self, signal, reference):
        self.cca.fit(signal, reference)
        res_x, res_y = self.cca.transform(signal, reference)
        corr = np.corrcoef(res_x.T, res_y.T)[0][1]
        return corr

    def getResults(self, sample, frequencies=None):
        if frequencies:         # if frequencies are provided, get reference signals
            self.getReferenceSignals(len(sample), frequencies)
        return [self.getCorr(sample, signal.T) for signal in self.reference_signals]

    def getWindowResults(self,sample_ep, frequencies, corr_freq, noise_template=None, add=False):
        """ Compute the accuracy of SSVEP detection using CCA
            
        Parameters
        ----------
        sample_ep : 
            3D array with shape (n_windows, n_channels, n_samples)
        frequencies : 
            list of frequencies at which we detect SSVEP
        corr_freq : 
            known SSVEP frequency of sample
        template : 
            noise template to add to CCA template
        add :
            whether or not to sum the harmonics in the original template
        
        Returns
        -------
        predictions :
            list of length n_frequencies containing proportion of windows predicted of certain frequency
        accuracy :
            decimal representing accuracy of SSVEP
        all_results:
            2D array with shape (n_windows, n_frequencies) containing CCA output
        
        """   
        self.getReferenceSignals(sample_ep.shape[-1], frequencies)
        if add:
            self.reference_signals = np.concatenate(
                                                    (np.sum(self.reference_signals[:,::2], axis=1,keepdims=True), 
                                                    np.sum(self.reference_signals[:,1::2], axis=1,keepdims=True)), axis=1)
        # add noise template to template if requested
        if noise_template is not None:
            for i, ref in enumerate(self.reference_signals):
                self.reference_signals[i] = np.array([temp + noise_template for temp in ref])
        right, wrong = 0,0
        all_results = []
        predictions = np.zeros(np.array(frequencies).shape)
        
        # compute CCA for each window
        for window in sample_ep:
            results = self.getResults(window.T)
            all_results.append(results)
            predictions[np.argmax(results)] += 1
            pred = frequencies[np.argmax(results)]
            if (pred == corr_freq):
                right +=1
            else:
                wrong +=1
        all_results = np.array(all_results)
        print(np.mean(all_results, axis=0))
        print(right+wrong)
        return list(predictions/(right+wrong)), right/(right + wrong), all_results
        
