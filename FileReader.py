# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 20:56:51 2018

@author: Marley
"""
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import mne

class FileReader():
    '''
    def bandpass_channels(self, arr, lowcut,highcut,order):
        nyq = 0.5 * self.fs
        b, a = signal.butter(order, [lowcut/nyq, highcut/nyq], btype='band')
        arr = signal.lfilter(b, a, arr, axis=-1)
        return arr
        '''
    def bandpass(self,data,start,stop):
        bp_Hz = np.zeros(0)
        bp_Hz = np.array([start,stop])
        b, a = signal.butter(3, bp_Hz/(self.fs / 2.0),'bandpass')
        print("Bandpass filtering to: " + str(bp_Hz[0]) + "-" + str(bp_Hz[1]) + " Hz")
        return signal.lfilter(b, a, data, 0)
    def filter_channel(self, ch,notch,highpass,bandpass, fs_Hz):
        if highpass:
            hp_cutoff_Hz = 1.0
            print("Highpass filtering at: " + str(hp_cutoff_Hz) + " Hz")
            b, a = signal.butter(2, hp_cutoff_Hz/(fs_Hz / 2.0), 'highpass')
            ch = signal.lfilter(b, a, ch, 0)
        notch_freq_Hz = np.array([60 * i for i in range(1, int(fs_Hz/2/60)+1)])  # main + harmonic frequencies
        if notch:
            for freq_Hz in np.nditer(notch_freq_Hz):  # loop over each target freq
                bp_stop_Hz = freq_Hz + 3.0*np.array([-1, 1])  # set the stop band
                b, a = signal.butter(3, bp_stop_Hz/(fs_Hz / 2.0), 'bandstop')
                ch = signal.lfilter(b, a, ch, 0)
                print("Notch filter removing: " + str(bp_stop_Hz[0]) + "-" + str(bp_stop_Hz[1]) + " Hz")
        if bandpass:
            ch = self.bandpass(ch, 1, 50)
        return ch
        
    def filter_channels(self, data,notch,highpass,bandpass, fs):
        ndata = []
        for ch in data:
            y = self.filter_channel(ch,notch,highpass,bandpass, fs)
            ndata.append(y)
        return np.array(ndata)
        
    def get_filename(self, case, trial=7, session=1):
        path = 'C:/Users/Christopher/Marley/BCI/OpenBCI_GUI/OpenBCI_GUI/SavedData/'
        path2 = 'C:/Users/Christopher/Marley/OpenBCI/OpenBCI_GUI/SavedData/'
        if trial == 3:
            if case == 6.66:
                return path2 + 'OpenBCI-RAW-2018-03-14_13-32-02.txt' # 6.6 attempt 3 light
            elif case == 7.5:
                return path2 + 'OpenBCI-RAW-2018-03-14_13-36-36.txt' #7.5 attempt 3 light
            elif case == 8.57:
                return path2 + 'OpenBCI-RAW-2018-03-14_13-44-46.txt' # 8.57 attempt 3 light
            elif case == 10:
                return path2 + 'OpenBCI-RAW-2018-03-14_13-49-40.txt'
                #return path2 + 'OpenBCI-RAW-2018-03-14_13-38-03.txt' # 10 attempt 3 light
            elif case == 'Baseline':
                return path2 + 'OpenBCI-RAW-2018-03-14_13-43-09.txt' # Baseline attempt 3 light
        if trial == 2:
            if case == 6.66:
                return path + 'OpenBCI-RAW-2018-03-08_21-38-54.txt' #6.6 Marley
            elif case == 7.52:
                return path2 + 'OpenBCI-RAW-2018-03-11_22-46-13.txt' # 7.5 attempt 2 dark
            elif case == 7.5:
                return path2 + 'OpenBCI-RAW-2018-03-11_22-54-19.txt' # 7.5 attempt 2 light
            elif case == 8.57:
                return path + 'OpenBCI-RAW-2018-03-09_17-12-40.txt' 
            elif case == 10:
                return path2 + 'OpenBCI-RAW-2018-03-11_22-55-18.txt' # 10 light
            elif case == 'Baseline':
                return 'C:/Users/Christopher/Marley/OpenBCI/OpenBCI_GUI/SavedData/OpenBCI-RAW-2018-03-12_11-37-33.txt'
        if trial == 4:
            if case == 7.5:
                return path + 'OpenBCI-RAW-2018-03-27_12-20-11.txt' #7.5 again
            if case == 12:
                return path + 'OpenBCI-RAW-2018-03-27_12-29-34.txt' #12
            if case == 10:
                return path + 'OpenBCI-RAW-2018-03-27_12-31-31.txt' #10
        if trial == 5: # Marley, gold cup electrodes, april 2
            if case == 7:
                return path + 'OpenBCI-RAW-2018-04-02_23-08-40.txt'
            if case == 7.52:
                return path + 'OpenBCI-RAW-2018-04-02_23-04-13.txt'
            if case == 7.51:
                return path + 'OpenBCI-RAW-2018-04-02_22-48-59.txt'
            if case == 7.5:
                return path + 'OpenBCI-RAW-2018-04-02_22-42-44.txt'
            if case == 10:
                return path + 'OpenBCI-RAW-2018-04-02_22-41-33.txt'
            if case == 8.57:
                return path + 'OpenBCI-RAW-2018-04-02_22-45-57.txt'
            if case == 12.1:
                return path + 'OpenBCI-RAW-2018-04-02_23-11-48.txt'
            if case == 12:
                return path + 'OpenBCI-RAW-2018-04-02_22-47-06.txt'
            if case == 'Baseline':
                return path + 'OpenBCI-RAW-2018-04-02_22-40-21.txt'
            if case == 'baseline':
                return path + 'OpenBCI-RAW-2018-04-02_22-44-29.txt'
        if trial == 6: #Marley gold cup electrodes april 8
            if session == 2: # all channels on
                if case == 'BaselineEnd':
                    return path + 'OpenBCI-RAW-2018-04-08_13-03-36.txt'
                if case == 10:
                    return path + 'OpenBCI-RAW-2018-04-08_13-01-43.txt'
                if case == 12:
                    return path + 'OpenBCI-RAW-2018-04-08_12-59-14.txt'
                if case == 8:
                    return path + 'OpenBCI-RAW-2018-04-08_12-56-05.txt'
                if case == 'Baseline':
                    return path + 'OpenBCI-RAW-2018-04-08_12-52-20.txt'
                if case == 'ec':
                    return path + 'OpenBCI-RAW-2018-04-08_13-37-53.txt'
                if case == 'BaselineS':
                    return path + 'OpenBCI-RAW-2018-04-08_13-58-00.txt'
            else: # turn off channels
                if case == 'Baseline':
                    return path + 'OpenBCI-RAW-2018-04-08_12-51-06.txt'
                if case == 'BaselineEnd':
                    return path + 'OpenBCI-RAW-2018-04-08_13-02-35.txt' #baseline taken at the end
                if case == 8:
                    return path + 'OpenBCI-RAW-2018-04-08_12-54-41.txt'
                if case == 12.1:
                    return path + 'OpenBCI-RAW-2018-04-08_12-56-55.txt'
                if case == 12:
                    return path + 'OpenBCI-RAW-2018-04-08_14-28-37.txt'
                if case == 10:
                    return path + 'OpenBCI-RAW-2018-04-08_13-00-15.txt'
                if case== 'ec':
                    return path + 'OpenBCI-RAW-2018-04-08_13-38-42.txt'
                if case== 'BaselineS':
                    return path + 'OpenBCI-RAW-2018-04-08_13-45-47.txt' #separate baseline
                if case =='BaselineS2':
                    return path + 'OpenBCI-RAW-2018-04-08_14-02-30.txt'
        if trial == 7:
            if case == 'Baseline':
                return path + 'OpenBCI-RAW-2018-04-17_17-11-59.txt'
            if case == 'Baseline2':
                return path + 'OpenBCI-RAW-2018-04-17_17-13-22.txt'
            if case == 8:
                return path + 'OpenBCI-RAW-2018-04-17_17-14-24.txt'
            if case == 10:
                return path + 'OpenBCI-RAW-2018-04-17_17-17-39.txt'
            if case == 12:
                return path + 'OpenBCI-RAW-2018-04-17_20-58-19.txt'
                #return path + 'OpenBCI-RAW-2018-04-17_20-59-46.txt'
                #return path + 'OpenBCI-RAW-2018-04-17_21-01-10.txt'
                #return path + 'OpenBCI-RAW-2018-04-17_17-43-51.txt'
                #return path + 'OpenBCI-RAW-2018-04-17_17-41-31.txt'
                #return path + 'OpenBCI-RAW-2018-04-17_17-42-28.txt'
                #return path + 'OpenBCI-RAW-2018-04-17_17-19-49.txt'
            if case == 'BaselineEnd':
                return path + 'OpenBCI-RAW-2018-04-17_17-21-25.txt'
        elif case == 7.51:
            return path + 'OpenBCI-RAW-2018-03-08_22-20-12.txt' #7.5 Marley
        elif case == 3:
            return path + 'OpenBCI-RAW-2018-03-09_10-25-54.txt' #7.5 Marley
        elif case == 8.572:
            return path2 + 'OpenBCI-RAW-2018-03-11_22-41-42.txt'
        elif case == 'Baseline':
            return 'C:/Users/Christopher/Marley/OpenBCI/OpenBCI_GUI/SavedData/OpenBCI-RAW-2018-03-12_11-37-33.txt'
            
    def get_enobio_fname(self, case, trial=2,session=1):
        if trial ==1:
            if case == 7.5:
                return '20180404182031_Patient01.easy'
            if case == 8.57:
                return '20180404182323_Patient01.easy'
            if case == 10:
                return '20180404181747_Patient01.easy'
            if case == 12:
                return '20180404181025_Patient01.easy'
            if case == 12.1:
                return '20180404181535_Patient01.easy'
            if case == 'Baseline':
                return '20180404181213_Patient01.easy'
            if case == 'Baseline2':
                return '20180404182147_Patient01.easy'
        if trial==2:
            if session == 1:
                if case == 10:
                    return '20180409172119_Patient01.easy'
                if case == 8:
                    return '20180409172002_Patient01.easy'
                if case == 12:
                    return '20180409171848_Patient01.easy'
            if case == 'Baseline':
                return '20180409171754_Patient01.easy'
            if case == 'BaselineEnd':
                return '20180409172209_Patient01.easy'
            if session == 2:
                if case == 8:
                    return '20180409172614_Patient01.easy'
                if case == 12:
                    return '20180409172515_Patient01.easy'
                if case == 10:
                    return '20180409172410_Patient01.easy'
            if case == 'BaselineEnd2':
                return '20180409172703_Patient01.easy'
            if case == 'BaselineEnd3':
                return '20180409172747_Patient01.easy'
    def get_data(self, HEADSET, filter_, notch=True, highpass=True,bandpass=True, cutoff=5,limit=None, corr_freq=0, plot=False, session=1):
        if HEADSET == 'AVI':
            self.fs = 512
            data = np.loadtxt('../BCI/AVI_SSVEP_Dataset_CSV/single/Sub1_singletarget_EEG.dat',delimiter=',', dtype=float)
            n = 3
            dic = [10,10,10,12,12,12,6.5,6.5,6.5,6,6,6,6,6,6,7.5,7.5,7.5,7,7,7,8.2,8.2,8.2,9.3,9.3,9.3]
            corr_freq = dic[n]
            sample = data[:, n:n+3].T # 10 hz
            print(sample.shape)
        elif HEADSET == 'OBCI':
            self.fs = 250
            fname = self.get_filename(corr_freq, session=session)
                
            # load data into numpy array
            #data = np.loadtxt(fname,
            #                  delimiter=',',).T
            sample = np.loadtxt(fname,
                      delimiter=',',
                      skiprows=7,
                      usecols=(1,2)).T
            
            '''draw_specgram(data[0])
            draw_specgram(bdata[0])
            draw_specgram(ndata[0])
            '''
            #data = signal.detrend(data.T).T
            #sample = data[:LENGTH]
            #sample = ch.T[:LENGTH]
            #plt.plot(sample[:,0])
        elif HEADSET == 'Enobio':
            #path = 'C:/Users/Christopher/Marley/BCI/eeglab/SSVEP_Data/'
            #fname = 'Liviu' + str(corr_freq) + '.easy'
            path = 'C:/Users/Christopher/Documents/NIC/'
            fname = self.get_enobio_fname(corr_freq, session=session)
            data = np.loadtxt(path + fname).T
            sample = data[0:2]
            self.fs = 500
        elif HEADSET == 'Biosemi':
            path = 'C:/Users/Christopher/Marley/BCI/eeglab/SSVEP_Data/'
            fname = 'Liviu' + str(corr_freq) + '.bdf'
            #if corr_freq = 6.66:
                #fname = 'Liviu6.6Dark.bdf'
            raw = mne.io.read_raw_edf(path + fname)
            data, time = raw[:]
            sample = np.array(data[[26,27,29]])
            self.fs= 2048
        elif HEADSET == 'Epoc':
            path = 'C:/Users/Christopher/Marley/BCI/eeglab/SSVEP_Data/'
            if corr_freq == 7.5:
                fname = 'marley-10-15.03.18.13.39.41.edf'
            if corr_freq == 6.66:
                fname = 'marley-6.6-15.03.18.13.43.19.edf'
            if corr_freq == 10:
                fname = 'marley-10real-15.03.18.13.41.36.edf'
            if corr_freq == 'Baseline':
                fname = 'marley-baseline-15.03.18.13.44.32.edf'
            raw = mne.io.read_raw_edf(path + fname)
            data, time = raw[:]
            sample = np.array(data[[9,10]])
            self.fs = 128
        if filter_:
            sample = self.filter_channels(sample, notch, highpass,bandpass, self.fs)
            '''
            if bandpass:
                sample = self.bandpass_channels(sample, 1, 50, 5)
                '''
        sample = sample[:,self.fs*cutoff:-self.fs*cutoff]
        if limit is not None:
            sample = sample[:,:limit]
        if plot:    
            plt.plot(sample[0])
        print(sample.shape)
        return sample