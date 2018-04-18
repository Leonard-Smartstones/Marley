# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:10:19 2018

@author: Marley
"""

import numpy as np
from scipy.signal import butter, lfilter, welch
from scipy.stats import zscore, norm
from sklearn.base import BaseEstimator, TransformerMixin


class Filterer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 active=True,
                 id=0,
                 epoch=3,
                 filter_width=1.,
                 nb_chan=2,
                 pred_freq=25,
                 sample_rate=250,
                 ssvep_freq=6,
                 ssvep_range_high=60,
                 ssvep_range_low=6):
        self.active = active
        self.count_ = 0
        self.epoch_in_samples = int(sample_rate * epoch)
        self.nb_chan = nb_chan
        self.id = id
        self.sample_rate = sample_rate
        self.filter_width = filter_width
        self.filter_high = ssvep_freq + filter_width
        self.filter_low = ssvep_freq - filter_width
        self.pred_freq = pred_freq
        self.ssvep_freq = ssvep_freq
        self.ssvep_range_high = ssvep_range_high
        self.ssvep_range_low = ssvep_range_low

    def pred_time(self):
        """
        Increments local counter and checks against pred_freq. If self._count is hit, then reset counter and return
            true, else return false.
        :return:
        """
        if not self.active:
            return False
        self.count_ += 1
        if self.count_ >= self.pred_freq:
            self.count_ = 0
            return True
        return False

    def predict_proba(self, X):
        """
        Return a probability between 0 and 1
        :param X: (array like)
        :return:
        """
        # First we take a welch to decompose the new epoch into fequency and power domains
        freq, psd = welch(X, int(self.sample_rate), nperseg=4096)

        # Then normalize the power.
        # Power follows chi-square distribution, that can be pseudo-normalized by a log (because chi square
        #   is aproximately a log-normal distribution)
        psd = np.log(psd)
        psd = np.mean(psd, axis=0)

        # Next we get the index of the bin we are interested in
        low_index = np.where(freq > self.filter_low)[0][0]
        high_index = np.where(freq < self.filter_high)[0][-1]

        # Then we find the standard deviation of the psd over all bins between range low and high
        low_ssvep_index = np.where(freq >= self.ssvep_range_low)[0][0]
        high_ssvep_index = np.where(freq <= self.ssvep_range_high)[0][-1]

        zscores = np.zeros(psd.shape)
        zscores[low_ssvep_index:high_ssvep_index] = zscore(psd[low_ssvep_index:high_ssvep_index])

        pred = norm.cdf(zscores[low_index:high_index+1].mean())

        if np.isnan(pred):
            return 0.0
        else:
            return pred