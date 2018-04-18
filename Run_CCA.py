# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:32:47 2018

@author: Marley
"""

import numpy as np
from numpy import fft
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from CCA import CcaExtraction
from FileReader import FileReader
import csv
import itertools
import sys
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
sys.path.append('C://Users//Christopher//Marley//BCI//amplitude-modulation-analysis-module')
from am_analysis import am_analysis as ama

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
def plot_average_spectrum(data, window_length, offset, fig=False):
    if fig:
        plt.figure()
    data_ep = np.squeeze(epoch_data(data, window_length, offset))
    psd = ama.rfft_psd(data_ep.T,HEADSET_FREQ)
    m = np.mean(psd['PSD'], axis=-1)
    mpsd = psd
    mpsd['PSD'] = np.reshape(m, [-1,1])
    ama.plot_psd_data(mpsd, f_range=np.array([0,60]))
def get_spectrum(data):
    psd = ama.rfft_psd(data,HEADSET_FREQ)
    #m = np.mean(psd['PSD'], axis=-1)
    #mpsd = psd
    #mpsd['PSD'] = np.reshape(m, [-1,1])
    ama.plot_psd_data(psd, f_range=np.array([0,60]))
'''
def get_template(data):
    a = fft.ifft(fft.fft(data))
    a = fft.ifft(np.mean(np.array([fft.fft(d) for d in data]), axis=0))
        
    #a = scipy.signal.detrend(a)
    return sklearn.preprocessing.scale(a).real
'''
def plot_fft(f, amp=True, fig=True):
    if fig:
        plt.figure()
    n = len(f)
    freq_ax = np.fft.fftfreq(n, 1/HEADSET_FREQ)[range(int(n/2))]
    Y = f[range(int(n/2))]
    if amp:
        Y = np.abs(Y)
    plt.plot(freq_ax, Y)
    
def get_template(data_ep, amp=False, plot_flag=True, scale=True):
    f = fft.fft(data_ep, axis=-1)
    if amp:
        f = np.abs(f)
    f = np.mean(f, axis =0)
    template = fft.ifft(f)
    if scale:
        scaler = MinMaxScaler(feature_range=[-1,1])
        template = np.squeeze(scaler.fit_transform(template.reshape(-1,1)))
        #template = sklearn.preprocessing.scale(template)
    if plot_flag:
        plot_fft(f)
        #ym = np.max(Y[int(n/10):]) * 2
        #plt.ylim(ymax=ym)
    return template
def epoch_data(data, window_length, offset):
    arr = []
    i = 0
    while i + window_length < data.shape[-1]:
        arr.append(data[:,i:i+window_length])
        i += offset
    return np.array(arr)
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y
def bandpass_channels(data, lowcut, highcut, fs, order=5):
    bdata = []
    for ch in data:
        for i in range (0, order):
            ch = butter_bandpass_filter(ch, lowcut, highcut, fs, order=1)
        bdata.append(ch)
    return np.array(bdata)
    
def draw_specgram(ch, fs_Hz):
    NFFT = fs_Hz*2
    overlap = NFFT - int(0.25 * fs_Hz)
    spec_PSDperHz, spec_freqs, spec_t = mlab.specgram(np.squeeze(ch),
                                   NFFT=NFFT,
                                   window=mlab.window_hanning,
                                   Fs=fs_Hz,
                                   noverlap=overlap
                                   ) # returns PSD power per Hz
    # convert the units of the spectral data
    
    spec_PSDperBin = spec_PSDperHz * fs_Hz / float(NFFT)
    f_lim_Hz = [0, 250]   # frequency limits for plotting
    plt.figure(figsize=(10,5))
    ax = plt.subplot(1,1,1)
    plt.pcolor(spec_t, spec_freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
    #plt.clim([-25,26])
    plt.xlim(spec_t[0], spec_t[-1]+1)
    plt.ylim(f_lim_Hz)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.show()
    
def write_csv(name, obj):
    with open(name, "w") as f:
        writer = csv.writer(f)
        writer.writerows(obj)

def plot_confusion_matrix(cm, frequencies, title='Confusion matrix', cmap = plt.cm.Greys):
    plt.matshow(cm, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, frequencies, rotation=45)
    plt.yticks(tick_marks, frequencies)
    fmt = '.2f' #if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #plt.tight_layout()
    plt.ylabel("actual")
    plt.xlabel("predicted")
def plot_cumulative_accuracy(all_resu, frequencies):
    plt.figure()
    plt.ylim(0,1.1)
    for corr_freq, result in zip(frequencies, all_resu):
        idx = frequencies.index(corr_freq)
        pred = np.argmax(np.squeeze(result), axis = -1)
        num_corr = 0
        num = 0
        arr = []
        for i in pred:
            if i == idx:
                num_corr +=1
            num += 1
            arr.append(num_corr/num)
        plt.plot(arr, label=corr_freq)
    plt.legend()
    
    
    
HEADSET = 'OBCI'
filter_ =1
highpass = 0
notch = 1
bandpass=1
plot_flag = 1
#frequencies = [7.5,10, 12]
frequencies = [8,10,12]
#frequencies = [6.66, 7.5, 10]
corr_frequencies = frequencies
window_lengths = [4]
all_predictions = []
all_acc = []
all_results = []
limit = 7386


fio = FileReader()
sampleb = fio.get_data(HEADSET, filter_, cutoff=5, notch=notch, highpass=highpass,bandpass=bandpass, corr_freq='Baseline', session=1)
HEADSET_FREQ = fio.fs
window_length = 4*HEADSET_FREQ
pred_interval = int(HEADSET_FREQ/2)
sampleb = sampleb[[0]]
av = epoch_data(sampleb, window_length, pred_interval)
#template = av[0,0]
template = get_template(av[:,0,:],scale=False,amp=True, plot_flag=plot_flag)/2
template = None
plt.figure()
plot_average_spectrum(sampleb, window_length, pred_interval)

curr = [['Filename','Filter','Window Length','Prediction Frequencies','Proportion Predicted','Accuracy']]
for corr_freq in corr_frequencies:
    #plt.figure(figsize=(10,5))
    #ax = plt.subplot(1,1,1)
    sample = fio.get_data(HEADSET, filter_, cutoff=1, limit=limit,notch=notch, highpass=highpass, bandpass=bandpass,corr_freq=corr_freq,session=1)
    sample = sample[[0]]
    a = CcaExtraction(HEADSET_FREQ)
    accuracies = []
    for num in window_lengths:
        window_length = int(HEADSET_FREQ * num)
        pred_interval = int(HEADSET_FREQ/2)
        plot_average_spectrum(sample, window_length, pred_interval)
        sample_ep = epoch_data(sample, window_length, pred_interval)
        predictions, acc, result = a.getWindowResults(sample_ep, frequencies, corr_freq, window_length,template=template,add=False)
        all_results.append(result)
        accuracies.append(acc)
        all_predictions.append(predictions)
        #print(a.getResults(sample.T, frequencies=frequencies))
        
        #result = [filter_, str(window_length/HEADSET_FREQ) + ' sec', frequencies, predictions, acc]
        #print(result)
    all_acc.append(accuracies)
    curr.append(result)
plot_confusion_matrix(np.array(all_predictions),corr_frequencies)
plot_cumulative_accuracy(all_results, corr_frequencies)
#plt.plot(av)
#plt.figure()
#plt.plot(a.reference_signals[0,0])
if len(window_lengths) > 1:
    plt.figure()
    for acc in all_acc:
        plt.plot(window_lengths, acc)
    
'''
arr = []
for i in np.arange(250, limit, step=250):
    arr.append(a.getResults(sample.T[:i], frequencies=corr_frequencies))
    '''