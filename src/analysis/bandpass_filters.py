"""Butterworth filter-bank definitions shared by the EEG workflows."""

from scipy.signal import butter, sosfiltfilt


BANDS = [(max(1, low), low + 2) for low in range(0, 60, 2)]


def band_filter(band, sfreq):
    return butter(4, band, btype="bandpass", fs=sfreq, output="sos")


def filter_band(data, band, sfreq):
    return sosfiltfilt(band_filter(band, sfreq), data)
