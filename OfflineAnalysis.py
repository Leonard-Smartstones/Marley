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
from SNR import SNR
from Benchmark import Benchmark
from FileReader import FileReader
import itertools
import sys
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
sys.path.append('C://Users//Christopher//Marley//BCI//amplitude-modulation-analysis-module')
from am_analysis import am_analysis as ama

def plot_average_psd(data, window_length, offset, fig=False):
    if fig:
        plt.figure()
    data_ep = np.squeeze(epoch_data(data, window_length, offset))
    psd = ama.rfft_psd(data_ep.T,fs)
    m = np.mean(psd['PSD'], axis=-1)
    mpsd = psd
    mpsd['PSD'] = np.reshape(m, [-1,1])
    ama.plot_psd_data(mpsd, f_range=np.array([0,60]))
def plot_psd(data):
    psd = ama.rfft_psd(data,fs)
    ama.plot_psd_data(psd, f_range=np.array([0,60]))
def plot_fft(f, amp=True, fig=True):
    if fig:
        plt.figure()
    n = len(f)
    freq_ax = np.fft.fftfreq(n, 1/fs)[range(int(n/2))]
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
    if plot_flag:
        plot_fft(f)
        #ym = np.max(Y[int(n/10):]) * 2
        #plt.ylim(ymax=ym)
    return template
def epoch_data(data, window_length, offset):
    arr = []
    i = 0
    while i + window_length < data.shape[-1]:
        arr.append(data.T[i:i+window_length].T)
        i += offset
    return np.array(arr)
    
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
    

def plot_confusion_matrix(cm, corr_frequencies, frequencies, title='Confusion matrix', cmap = plt.cm.Greys):
    plt.matshow(cm, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, corr_frequencies, rotation=45)
    plt.yticks(tick_marks, frequencies)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #plt.tight_layout()
    plt.ylabel("actual")
    plt.xlabel("predicted")
def plot_cumulative_accuracies(arr, frequencies):
    if arr.ndim > 3:
        for i in range (0, arr.shape[2]):
            plot_cumulative_accuracy(arr[:,:,i], frequencies)
    else:
        plot_cumulative_accuracy(arr, frequencies)
        
def plot_cumulative_accuracy(outputs, frequencies):
    plt.figure()
    plt.ylim(0,1.1)
    for corr_freq, output in zip(frequencies, outputs):
        idx = frequencies.index(corr_freq)
        pred = np.argmax(np.squeeze(output), axis = -1)
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
    
def plot_acc_by_window_len(window_lengths, arr):
    plt.figure()
    plt.ylim(0,1.1)
    for acc in arr:
        plt.plot(window_lengths, acc)
    
fs_dict = {'openbci': 250, 'enobio': 500, 'wd': 300, 'epoc':128, 'biosemi':2048}
HEADSET = 'openbci'
filter_ =1
highpass = 0
notch = 1
bandpass=1
plot_flag = 1
trial = 7
#corr_frequencies = [7.5,10, 12]
#corr_frequencies = [6.66, 7.5, 10]
corr_frequencies = [8,10,12]
frequencies = corr_frequencies
window_lengths = [4]
recording_length = 30
pred_interval = 0.5


fs = fs_dict[HEADSET]
classifier = Benchmark(fs)
pred_interval = int(fs * pred_interval)

# Load data from files
data = {}
fio = FileReader()
baseline = fio.get_data(HEADSET, filter_, cutoff=1, notch=notch, highpass=highpass,bandpass=bandpass, corr_freq='Baseline', trial=trial, session=1)
data['Baseline'] = baseline
for corr_freq in corr_frequencies:
    sample = fio.get_data(HEADSET, filter_, cutoff=1, limit=recording_length*fs,notch=notch, highpass=highpass, bandpass=bandpass,corr_freq=corr_freq,trial=trial, session=1)
    data[corr_freq] = sample

# Plot averaged psd for each channel, for each frequency
for ch_idx in range (0, len(sample)):
    plt.figure()
    for sample in data.values():
        plot_average_psd(sample[ch_idx], int(4 * fs), pred_interval)

# Compute predictions and prediction accuracy for each window length, for each data sample
acc_by_window_len  = []
for window_length in window_lengths:
    window_size = int(fs * window_length)
    accuracies_set = []
    output_set = []
    prediction_set = []
    av = epoch_data(baseline[[0]], window_size, pred_interval)
    template = get_template(av[:,0,:],scale=False,amp=True, plot_flag=plot_flag)/2
    for corr_freq in corr_frequencies:        
        sample = data[corr_freq]
        sample_ep = epoch_data(sample, window_size, pred_interval)
        predictions, acc, outputs = classifier.getWindowResults(sample_ep, frequencies, corr_freq)
        prediction_set.append(predictions)
        accuracies_set.append(acc)
        output_set.append(outputs)
        
    acc_by_window_len.append(accuracies_set)

if plot_flag:
    plot_confusion_matrix(np.array(prediction_set),corr_frequencies, frequencies)
    plot_cumulative_accuracies(np.array(output_set), corr_frequencies)
    if len(window_lengths) > 1:
        plot_acc_by_window_len(window_lengths, acc_by_window_len)