# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:14:41 2018

@author: Marley
"""
import numpy as np
from filterer import Filterer

    
def getWindowResults(sample_ep, sample_rate, frequencies, corr_freq):
        right, wrong = 0,0
        filters = [Filterer(filter_width=0.5,
                            ssvep_freq=freq,
                            sample_rate=sample_rate)
                        for freq in frequencies]
        predictions = np.zeros(np.array(frequencies).shape)
        for window in sample_ep:
            results = [filt.predict_proba(window) for filt in filters]
            print(results)
            predictions[np.argmax(results)] += 1
            pred = frequencies[np.argmax(results)]
            #print(pred)
            if (pred == corr_freq):
                right +=1
            else:
                wrong +=1
        print(right+wrong)
        return list(predictions/(right+wrong)), right/(right + wrong)
filter_ = 0
#curr3 = [['Filename','Filter','Window Length','Prediction Frequencies','Proportion Predicted','Accuracy']]


#freq = [10]
frequencies =[6.66, 7.5,10]

'''
a = CcaExtraction(250)
ref = a.getReferenceSignals(500, freq)
sample = np.vstack((ref[0][[0]],ref[0][[0]]))

filters = [Filterer(epoch=5,
                    filter_width=0.5,
                    ssvep_freq=freq,
                    sample_rate=250) for freq in frequencies]
results = [filt.predict_proba(sample) for filt in filters]
pred = frequencies[np.argmax(results)]
print(results)
'''
HEADSET = 'Biosemi'
all_predictions = []
frequencies = [6.66,7.5,10]
corr_frequencies = [6.66, 7.5, 10]
for corr_freq in corr_frequencies:
    #plt.figure(figsize=(10,5))
    #ax = plt.subplot(1,1,1)
    sample = getData(HEADSET, filter_, corr_freq=corr_freq)
    window_length = HEADSET_FREQ * 4
    overlap = int(HEADSET_FREQ * 3.5)
    sample_ep, _,_ = ama.epoching(sample.T, window_length, overlap)
    
    predictions, acc = getWindowResults(sample_ep.T, HEADSET_FREQ, frequencies, corr_freq)
    all_predictions.append(predictions)
    result = [filter_, str(window_length/HEADSET_FREQ) + ' sec', frequencies, predictions, 'n/a']
    print(result)
    #curr3.append(result)
plot_confusion_matrix(np.array(all_predictions),corr_frequencies)