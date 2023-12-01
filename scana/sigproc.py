
import numpy
import scipy.signal

__all__ = ['SignalProcessor',
           'LowPassBesselFilter', 'LowPassButterFilter',
           'HighPassBesselFilter', 'HighPassBesselFilter',
           'BandStopBesselFilter', 'BandStopButterFilter']


class SignalProcessor:

    def apply(self, data):
        raise NotImplementedError('Error, method "SignalProcessor.apply(...)" not implemented!')


class BaseLineCorrection(SignalProcessor):

    def __init__(self, value):
        self._value = float(value)

    def apply(self, data):
        return data - self._value


class LowPassBesselFilter(SignalProcessor):

    def __init__(self, signal_nyq_freq, filter_order, filter_freq):
        signal_nyq_freq = float(signal_nyq_freq)
        filter_order = int(filter_order)
        filter_freq = float(filter_freq)
        critical_freq = filter_freq / signal_nyq_freq

        # create filter
        self._filter = scipy.signal.bessel(filter_order, critical_freq, output='ba',
                                           btype='lowpass', analog=False)
        self._pad_len = 3*max(len(self._filter[0]), len(self._filter[1]))

    def apply(self, data):
        return scipy.signal.filtfilt(self._filter[0], self._filter[1], data,
                                     method='pad', padtype='even', padlen=self._pad_len)


class LowPassButterFilter(SignalProcessor):

    def __init__(self, signal_nyq_freq, filter_order, filter_freq):
        signal_nyq_freq = float(signal_nyq_freq)
        filter_order = int(filter_order)
        filter_freq = float(filter_freq)
        critical_freq = filter_freq / signal_nyq_freq
        # create filter
        self._filter = scipy.signal.butter(filter_order, critical_freq, output='ba',
                                           btype='lowpass', analog=False)
        self._pad_len = 3*max(len(self._filter[0]), len(self._filter[1]))

    def apply(self, data):
        return scipy.signal.filtfilt(self._filter[0], self._filter[1], data,
                                     method='pad', padtype='even', padlen=self._pad_len)


class HighPassBesselFilter(SignalProcessor):

    def __init__(self, signal_nyq_freq, filter_order, filter_freq):
        signal_nyq_freq = float(signal_nyq_freq)
        filter_order = int(filter_order)
        filter_freq = float(filter_freq)
        critical_freq = filter_freq/signal_nyq_freq
        # create filter
        self._filter = scipy.signal.bessel(filter_order, critical_freq, output='ba',
                                           btype='highpass', analog=False)
        self._pad_len = 3*max(len(self._filter[0]), len(self._filter[1]))

    def apply(self, data):
        return scipy.signal.filtfilt(self._filter[0], self._filter[1], data,
                                     method='pad', padtype='even', padlen=self._pad_len)


class HighPassButterFilter(SignalProcessor):

    def __init__(self, signal_nyq_freq, filter_order, filter_freq):
        signal_nyq_freq = float(signal_nyq_freq)
        filter_order = int(filter_order)
        filter_freq = float(filter_freq)
        critical_freq = filter_freq/signal_nyq_freq
        # create filter
        self._filter = scipy.signal.butter(filter_order, critical_freq, output='ba',
                                           btype='highpass', analog=False)
        self._pad_len = 3*max(len(self._filter[0]), len(self._filter[1]))

    def apply(self, data):
        return scipy.signal.filtfilt(self._filter[0], self._filter[1], data,
                                     method='pad', padtype='even', padlen=self._padlen)


class BandStopBesselFilter(SignalProcessor):

    def __init__(self, signal_nyq_freq, filter_order, filter_freq_low, filter_freq_high):
        signal_nyq_freq = float(signal_nyq_freq)
        filter_order = int(filter_order)
        filter_freq_low = float(filter_freq_low)
        filter_freq_high = float(filter_freq_high)
        critical_freq = (filter_freq_low/signal_nyq_freq,
                         filter_freq_high/signal_nyq_freq)
        # create filter
        self._filter = scipy.signal.bessel(filter_order, critical_freq, output='ba',
                                           btype='bandstop', analog=False)
        self._pad_len = 3*max(len(self._filter[0]), len(self._filter[1]))

    def apply(self, data):
        return scipy.signal.filtfilt(self._filter[0], self._filter[1], data,
                                     method='pad', padtype='even', padlen=self._pad_len)


class BandStopButterFilter(SignalProcessor):

    def __init__(self, signal_nyq_freq, filter_order, filter_freq_low, filter_freq_high):
        signal_nyq_freq = float(signal_nyq_freq)
        filter_order = int(filter_order)
        filter_freq_low = float(filter_freq_low)
        filter_freq_high = float(filter_freq_high)
        critical_freq = (filter_freq_low/signal_nyq_freq,
                         filter_freq_high/signal_nyq_freq)
        # create filter
        self._filter = scipy.signal.butter(filter_order, critical_freq, output='ba',
                                           btype='bandstop', analog=False)
        self._pad_len = 3*max(len(self._filter[0]), len(self._filter[1]))

    def apply(self, data):
        return scipy.signal.filtfilt(self._filter[0], self._filter[1], data,
                                     method='pad', padtype='even', padlen=self._pad_len)
