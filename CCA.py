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

    def getReferenceSignal(self, target_reference, length):
        if self.short_signal:
            return np.array([target_reference[j][:length] for j in range(len(target_reference))])
        else:
            return np.array(target_reference)

    def getResults(self, sample, frequencies=None):
        if frequencies:         # if frequencies are provided, get reference signals
            self.getReferenceSignals(len(sample), frequencies)
        return [self.getCorr(sample, signal.T) for signal in self.reference_signals]
    '''def getResults(self, coordinates, length, target_freqs):
        return ((freq, self.getCorr(coordinates, self.getReferenceSignal(reference, length).T)) for freq, reference in zip(target_freqs, self.reference_signals))
        '''
    def getWindowResults(self,sample_ep, frequencies, corr_freq, window_length, template=None, add=False):
        self.getReferenceSignals(window_length, frequencies)
        if add:
            self.reference_signals = np.concatenate(
                                                    (np.sum(self.reference_signals[:,::2], axis=1,keepdims=True), 
                                                    np.sum(self.reference_signals[:,1::2], axis=1,keepdims=True)), axis=1)
        if template is not None:
            for i, ref in enumerate(self.reference_signals):
                self.reference_signals[i] = np.array([temp + template for temp in ref])
            #self.reference_signals = np.array([np.vstack((freq,template)) for freq in self.reference_signals])
        right, wrong = 0,0
        all_results = []
        predictions = np.zeros(np.array(frequencies).shape)
        for window in sample_ep:
            results = self.getResults(window.T)
            all_results.append(results)
            predictions[np.argmax(results)] += 1
            pred = frequencies[np.argmax(results)]
            #print(pred)
            if (pred == corr_freq):
                right +=1
            else:
                wrong +=1
        all_results = np.array(all_results)
        print(np.mean(all_results, axis=0))
        print(right+wrong)
        return list(predictions/(right+wrong)), right/(right + wrong), all_results
    def getRanking(self, results):
        return sorted(results, key=lambda x: x[1], reverse=True)
        
