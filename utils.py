import math
import numpy as np
import scipy.signal as sp_sig

from typing import List


def find_nearest_index(arr, val):
    idx = (np.abs(arr - val)).argmin()
    return idx


def fit_line(x1, y1, x2, y2):
    a = (y1 - y2) / (x1 - x2)
    b = y2 - a * x2
    return a, b


def align_pulse(t, data, a=None, b=None):
    if a is None and b is None:
        a, b = fit_line(t[0], data[0], t[-1], data[-1])
    pulse_offset = a * t + b
    return data - pulse_offset


def split_pulses(icp: np.ndarray, time: np.ndarray, fs: float) -> List[np.ndarray]:
    pulse_onsets = detect_pulses_in_signal(icp, fs)
    pulses = [icp[pulse_onsets[i]: pulse_onsets[i + 1]] for i in range(len(pulse_onsets) - 1)]
    times = [time[pulse_onsets[i]: pulse_onsets[i + 1]] for i in range(len(pulse_onsets) - 1)]

    return pulses, times, pulse_onsets


def detect_pulses_in_signal(x: np.ndarray, fs: float) -> np.ndarray:
    mean_x = np.nanmean(x)
    x[np.argwhere(np.isnan(x))] = mean_x
    
    dx = sp_sig.detrend(x)

    filter_critical_coefficient = 10
    filter_order = 6
    filter_max_ripple = 60
    filter_min_attenuation = 1
    filter_btype = 'lowpass'
    filter_ftype = 'cheby1'

    Wc1 = filter_critical_coefficient / fs

    b1, a1 = sp_sig.iirfilter( N=filter_order, Wn=Wc1, btype=filter_btype,
        rs=filter_max_ripple, rp=filter_min_attenuation, ftype=filter_ftype,)

    fx = sp_sig.filtfilt(b1, a1, dx)

    signal_peaks = detect_peaks_troughs(-fx, max_scale=fs)
    pulse_onset_inds = signal_peaks[:, 0]

    return pulse_onset_inds


def detect_peaks_troughs(x: np.ndarray, max_scale: float = 0) -> np.ndarray:
    N = len(x)
    if max_scale != 0:
        L = math.ceil(max_scale / 2) - 1
    else:
        L = math.ceil(N / 2) - 1

    mean_x = np.nanmean(x)
    x[np.argwhere(np.isnan(x))] = mean_x

    dx = sp_sig.detrend(x)

    Mx = np.zeros((N, L), dtype=np.float32)
    for kk in range(1, L + 1):
        # Last location skipped because of frequent anomalies
        right = dx[kk : -kk - 1] > dx[2 * kk : -1]
        left = dx[kk : -kk - 1] > dx[: -2 * kk - 1]
        Mx[kk : -kk - 1, kk - 1] = np.logical_and(left, right)

    dx = np.argmax(np.sum(Mx, 0))
    Mx = Mx[:, : dx + 1]

    _, col_count = Mx.shape
    Zx = col_count - np.count_nonzero(Mx, axis=1, keepdims=True)
    peaks = np.argwhere(Zx == 0)

    return peaks