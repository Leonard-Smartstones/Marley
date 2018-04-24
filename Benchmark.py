# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:14:41 2018

@author: Marley
"""
import numpy as np
from filterer import Filterer

class Benchmark():
    def __init__(self, HEADSET_FREQ):
        self.headset_freq = HEADSET_FREQ
        self.filters = None

    def getResults(self, window, frequencies=None):
        if frequencies:
            self.filters = [Filterer(filter_width=0.5,
                            ssvep_freq=freq,
                            sample_rate=self.headset_freq)
                        for freq in frequencies]
        return [filt.predict_proba(window) for filt in self.filters]
    
    def getWindowResults(self, sample_ep, frequencies, corr_freq):
        right, wrong = 0,0
        all_results = []
        self.filters = [Filterer(filter_width=0.5,
                            ssvep_freq=freq,
                            sample_rate=self.headset_freq)
                        for freq in frequencies]
        predictions = np.zeros(np.array(frequencies).shape)
        for window in sample_ep:
            results = self.getResults(window)
            all_results.append(results)
            predictions[np.argmax(results)] += 1
            pred = frequencies[np.argmax(results)]
            #print(pred)
            if (pred == corr_freq):
                right +=1
            else:
                wrong +=1
        print(right+wrong)
        return list(predictions/(right+wrong)), right/(right + wrong), all_results