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
    def bandpass(self,data,start,stop):
        bp_Hz = np.zeros(0)
        bp_Hz = np.array([start,stop])
        b, a = signal.butter(3, bp_Hz/(self.headset_frequency / 2.0),'bandpass')
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
        
    def get_openbci_filename(self, case, trial, session=1):
        path = 'C:/Users/Christopher/Marley/BCI/OpenBCI_GUI/OpenBCI_GUI/SavedData/'
        if trial == 1:
            path = 'Data/openbci/trial_1/'
            if case == 7.5:
                return path + 'OpenBCI-RAW-2018-03-11_22-46-13.txt' # 7.5 attempt 2 dark
            if case == 7.5:
                return path + 'OpenBCI-RAW-2018-03-11_22-54-19.txt' # 7.5 attempt 2 light
            if case == 8.57:
                return path + 'OpenBCI-RAW-2018-03-11_22-41-42.txt' # 8.57 attempt 2 light
            if case == 10:
                return path + 'OpenBCI-RAW-2018-03-11_22-55-18.txt' # 10 attempt 2 light
        if trial == 2:
            path = 'Data/openbci/trial_2/'
            if case == 6.66:
                return path + 'OpenBCI-RAW-2018-03-14_13-32-02.txt' # 6.6 attempt 3 light
            if case == 7.5:
                return path + 'OpenBCI-RAW-2018-03-14_13-36-36.txt' #7.5 attempt 3 light
            if case == 8.57:
                return path + 'OpenBCI-RAW-2018-03-14_13-44-46.txt' # 8.57 attempt 3 light
            if case == 10:
                return path + 'OpenBCI-RAW-2018-03-14_13-49-40.txt' # 10 attempt 3 light
            if case == 10:
                return path + 'OpenBCI-RAW-2018-03-14_13-38-03.txt' # 10 attempt 3 light
            if case == 'Baseline':
                return path + 'OpenBCI-RAW-2018-03-14_13-43-09.txt' # Baseline attempt 3 light
        if trial == 3:
            path = 'Data/openbci/trial_3/'
            if case == 7:
                return path + 'OpenBCI-RAW-2018-03-27_12-20-11.txt' #7
            if case == 12:
                return path + 'OpenBCI-RAW-2018-03-27_12-29-34.txt' #12
            if case == 10:
                return path + 'OpenBCI-RAW-2018-03-27_12-31-31.txt' #10
        if trial == 4: # Marley, gold cup electrodes, april 2
            path = 'Data/openbci/trial_4/'
            if case == 7:
                return path + 'OpenBCI-RAW-2018-04-02_23-08-40.txt'
            if case == 7.5:
                return path + 'OpenBCI-RAW-2018-04-02_22-42-44.txt'
            if case == 7.5:
                return path + 'OpenBCI-RAW-2018-04-02_22-48-59.txt'
            if case == 7.5:
                return path + 'OpenBCI-RAW-2018-04-02_23-04-13.txt'
            if case == 10:
                return path + 'OpenBCI-RAW-2018-04-02_22-41-33.txt'
            if case == 8.57:
                return path + 'OpenBCI-RAW-2018-04-02_22-45-57.txt'
            if case == 12:
                return path + 'OpenBCI-RAW-2018-04-02_22-47-06.txt'
            if case == 12:
                return path + 'OpenBCI-RAW-2018-04-02_23-11-48.txt'
            if case == 'Baseline':
                return path + 'OpenBCI-RAW-2018-04-02_22-40-21.txt'
            if case == 'Baseline':
                return path + 'OpenBCI-RAW-2018-04-02_22-44-29.txt'
        if trial == 5: # Marley gold cup electrodes april 8, all channels except for O1 turned off
            path = 'Data/openbci/trial_5/'
            if case == 'Baseline':
                return path + 'OpenBCI-RAW-2018-04-08_12-51-06.txt'
            if case == 'BaselineEnd':
                return path + 'OpenBCI-RAW-2018-04-08_13-02-35.txt' #baseline taken at the end
            if case == 8:
                return path + 'OpenBCI-RAW-2018-04-08_12-54-41.txt'
            if case == 12:
                return path + 'OpenBCI-RAW-2018-04-08_12-56-55.txt'
            if case == 12:
                return path + 'OpenBCI-RAW-2018-04-08_14-28-37.txt'
            if case == 10:
                return path + 'OpenBCI-RAW-2018-04-08_13-00-15.txt'
            if case== 'ec':
                return path + 'OpenBCI-RAW-2018-04-08_13-38-42.txt' # eyes closed
            if case== 'BaselineS':
                return path + 'OpenBCI-RAW-2018-04-08_13-45-47.txt' #separate baseline
            if case =='BaselineS2':
                return path + 'OpenBCI-RAW-2018-04-08_14-02-30.txt'
        if trial == 6: # Marley gold cup electrodes april 8
            path = 'Data/openbci/trial_6/'
            if case == 'Baseline':
                return path + 'OpenBCI-RAW-2018-04-08_13-02-35.txt'
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
                return path + 'OpenBCI-RAW-2018-04-08_13-37-53.txt' # eyes closed
            if case == 'BaselineS':
                return path + 'OpenBCI-RAW-2018-04-08_13-58-00.txt'
        if trial == 7:  # Wenya dry electrodes 
            path = 'Data/openbci/wenya_trial/'
            if case == 'Baseline':
                return path + 'OpenBCI-RAW-2018-04-17_17-11-59.txt'
            if case == 'Baseline2':
                return path + 'OpenBCI-RAW-2018-04-17_17-13-22.txt'
            if case == 8:
                return path + 'OpenBCI-RAW-2018-04-17_17-14-24.txt'
            if case == 10:
                return path + 'OpenBCI-RAW-2018-04-17_17-17-39.txt'
            if case == 12:
                return path + 'OpenBCI-RAW-2018-04-17_17-19-49.txt'
            if case == 'BaselineEnd':
                return path + 'OpenBCI-RAW-2018-04-17_17-21-25.txt'
        if trial == 8:
            path = 'Data/openbci/wenya_trial/'
            if case == 'Baseline':
                return path + 'OpenBCI-RAW-2018-04-17_17-43-51.txt'
            if case == 12:
                return path + 'OpenBCI-RAW-2018-04-17_17-41-31.txt'
            if case == 12:
                return path + 'OpenBCI-RAW-2018-04-17_17-42-28.txt'
        if trial == 9:
            path = 'Data/openbci/wenya_trial/'
            if case == 'Baseline':
                return path + 'OpenBCI-RAW-2018-04-17_20-58-19.txt' #baseline
            if case == 12:
                return path + 'OpenBCI-RAW-2018-04-17_20-59-46.txt'
            if case == 12:
                return path + 'OpenBCI-RAW-2018-04-17_21-01-10.txt'
        if trial == 10:
            path = 'Data/openbci/trial_10/'
            if case == 'Baseline':
                return path + 'OpenBCI-RAW-2018-04-26_14-37-23.txt'
            if case == 7:
                return path + 'OpenBCI-RAW-2018-04-26_14-45-33.txt'
                #return path + 'OpenBCI-RAW-2018-04-26_14-38-32.txt'
            if case == 8:
                return path + 'OpenBCI-RAW-2018-04-26_14-39-41.txt'
                #return path + 'OpenBCI-RAW-2018-04-26_14-33-35.txt'
            if case == 9:
                return path + 'OpenBCI-RAW-2018-04-26_14-36-08.txt'

        else:
            print('File not found')
            '''
        if case == 6.66:
            return path + 'OpenBCI-RAW-2018-03-08_21-38-54.txt' #6.6 Marley
        elif case == 8.57:
            return path + 'OpenBCI-RAW-2018-03-09_17-12-40.txt' 
        elif case == 7.51:
            return path + 'OpenBCI-RAW-2018-03-08_22-20-12.txt' #7.5 Marley
        elif case == 3:
            return path + 'OpenBCI-RAW-2018-03-09_10-25-54.txt' #7.5 Marley
        #elif case == 'Baseline':
            #return 'C:/Users/Christopher/Marley/OpenBCI/OpenBCI_GUI/SavedData/OpenBCI-RAW-2018-03-12_11-37-33.txt'
        '''
            
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
    def get_data(self, HEADSET, filter_, notch=True, highpass=True,bandpass=True, cutoff=0,limit=None, corr_freq=0, plot=False, trial=1,session=1):
        """ Load and preprocess data from single file
            
        Parameters
        ----------
        HEADSET : 
            string name of headset
        filter_ :
            whether or not to filter
        cutoff : 
            amount of samples to remove from beginning and end of sample
        limit : 
            maximum size of sample
        trial :
            which set to load
        
        Returns
        -------
        sample :
            2D array of shape (n_channels, n_samples)
        
        """
        if HEADSET == 'AVI':
            self.headset_frequency = 512
            data = np.loadtxt('../BCI/AVI_SSVEP_Dataset_CSV/single/Sub1_singletarget_EEG.dat',delimiter=',', dtype=float)
            n = 3
            dic = [10,10,10,12,12,12,6.5,6.5,6.5,6,6,6,6,6,6,7.5,7.5,7.5,7,7,7,8.2,8.2,8.2,9.3,9.3,9.3]
            corr_freq = dic[n]
            sample = data[:, n:n+3].T # 10 hz
            print(sample.shape)
        elif HEADSET == 'wd':
            self.headset_frequency = 300
            if trial == 1:
                path = 'Data/wearable_demo_trial1/'
            elif trial == 2:
                path = 'Data/wearable_demo_trial2/'
            '''
            if corr_freq == 8:
                fname = 'Marley_01_filtered.edf'
            if corr_freq == 12:
                fname = 'Marley_02_filtered.edf'
            if corr_freq == 10:
                fname = 'Marley_03_filtered.edf'
            if corr_freq == 'Baseline':
                fname = 'Marley_04_baseline_filtered.edf'
            '''
            
            if corr_freq == 8:
                fname = 'Marley_01_raw.edf'
            if corr_freq == 12:
                fname = 'Marley_02_raw.edf'
            if corr_freq == 10:
                fname = 'Marley_03_raw.edf'
            if corr_freq == 'Baseline':
                fname = 'Marley_04_baseline_raw.edf'
            raw = mne.io.read_raw_edf(path + fname,preload=True)
            data, time = raw[:]
            sample = np.array(data[[14,15]])
        elif HEADSET == 'openbci':
            self.headset_frequency = 250
            fname = self.get_openbci_filename(corr_freq, trial, session=session)
                
            sample = np.loadtxt(fname,
                      delimiter=',',
                      skiprows=7,
                      usecols=(1,2)).T
            sample = sample[[0]]
        elif HEADSET == 'enobio':
            self.headset_frequency = 500
            #path = 'C:/Users/Christopher/Marley/BCI/eeglab/SSVEP_Data/'
            #fname = 'Liviu' + str(corr_freq) + '.easy'
            path = 'C:/Users/Christopher/Documents/NIC/'
            fname = self.get_enobio_fname(corr_freq, session=session)
            data = np.loadtxt(path + fname).T
            sample = data[0:2]
        elif HEADSET == 'biosemi':
            path = 'C:/Users/Christopher/Marley/BCI/eeglab/SSVEP_Data/'
            fname = 'Liviu' + str(corr_freq) + '.bdf'
            #if corr_freq = 6.66:
                #fname = 'Liviu6.6Dark.bdf'
            raw = mne.io.read_raw_edf(path + fname)
            data, time = raw[:]
            sample = np.array(data[[26,27,29]])
            self.headset_frequency= 2048
        elif HEADSET == 'epoc':
            path = 'Data/epoc/'
            if corr_freq == 6.66:
                fname = 'marley-6.6-15.03.18.13.43.19.edf'
            if corr_freq == 7.5:
                fname = 'marley-7.5-15.03.18.13.39.41.edf'
            if corr_freq == 10:
                fname = 'marley-10-15.03.18.13.41.36.edf'
            if corr_freq == 'Baseline':
                fname = 'marley-baseline-15.03.18.13.44.32.edf'
            raw = mne.io.read_raw_edf(path + fname)
            data, time = raw[:]
            sample = np.array(data[[9,10]])
            self.headset_frequency = 128
        if filter_:
            sample = self.filter_channels(sample, notch, highpass,bandpass, self.headset_frequency)
            '''
            if bandpass:
                sample = self.bandpass_channels(sample, 1, 50, 5)
                '''
        sample = sample[:,self.headset_frequency*cutoff:sample.shape[-1] - self.headset_frequency*cutoff]
        if limit is not None:
            sample = sample[:,:limit]
        if plot:    
            plt.plot(sample[0])
        print(sample.shape)
        return sample