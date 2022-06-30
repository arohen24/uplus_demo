import math

import ahrs
import numpy as np
from ahrs import Quaternion
from scipy.signal import butter, lfilter, iirfilter

from uplus_demo.positioning.digitalfilter import LiveLFilter


# def butter_lowpass(cutoff, fs, order=5):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
#     return b, a
#
#
# def butter_lowpass_filter(data, cutoff, fs, order=5):
#     b, a = butter_lowpass(cutoff, fs, order=order)
#     y = lfilter(b, a, data)
#     return y


class PDRManager:
    def __init__(self, frequency=100, _G=9.80665):
        # define filter
        self.frequency = frequency
        order = 4
        cutoff = 2
        b, a = iirfilter(N=order, Wn=cutoff, fs=self.frequency, btype="low", ftype="butter")
        self.livefilter = LiveLFilter(b, a)

        self._G = _G
        self.last_q = [1., 0., 0., 0.]

        self.ts = np.empty(0)
        self.acc = np.empty((0, 3))
        self.gyro = np.empty((0, 3))

        self.acc_mag_filtered = np.empty(0)
        self.acc_filt_binary = np.empty(0)
        self.last_acc_filt_binary = 0
        self.heading_estimator = ahrs.filters.Madgwick(frequency=self.frequency, q0=self.last_q)

    def _set_acc_mag_filtered(self, ax_raw, ay_raw, az_raw):
        ax = np.asarray(ax_raw) * self._G
        ay = np.asarray(ay_raw) * self._G
        az = np.asarray(az_raw) * self._G

        sq_acc_mag = ax ** 2 + ay ** 2 + az ** 2
        sq_acc_mag.squeeze()

        acc_mag = np.sqrt(sq_acc_mag)
        acc_mag.squeeze()

        acc_mag_filtered = list()
        for val in acc_mag:
            acc_mag_filtered.append(self.livefilter(val))

        self.acc_mag_filtered = np.r_[self.acc_mag_filtered, acc_mag_filtered]

    def _get_acc_mag_filtered(self):
        acc_filt_detrend = self.acc_mag_filtered - self._G

        threshold = 0.4  # m/s2

        self.acc_filt_binary = [self.last_acc_filt_binary]
        excessive_wiggling = 5

        for i in range(0, len(acc_filt_detrend)):
            if threshold < acc_filt_detrend[i] < excessive_wiggling:
                self.acc_filt_binary.append(1)
            else:
                if acc_filt_detrend[i] < -threshold:
                    if self.acc_filt_binary[i - 1] == 1:
                        self.acc_filt_binary.append(0)
                    else:
                        self.acc_filt_binary.append(-1)
                else:
                    self.acc_filt_binary.append(0)

        self.acc_filt_binary = np.asarray(self.acc_filt_binary)
        return self.acc_mag_filtered

    def _get_stanceBegin_idx(self):
        window = math.ceil(0.4 * self.frequency)
        stanceBegin_idx = list()

        for i in range(window + 2, len(self.acc_filt_binary)):
            if self.acc_filt_binary[i] == -1 and self.acc_filt_binary[i - 1] == 0:
                if sum(self.acc_filt_binary[i - window:i - 2] > 0) > 1:
                    stanceBegin_idx.append(i)

        return np.asarray(stanceBegin_idx)

    def getStrideLength(self):
        acc_mag_filtered = self._get_acc_mag_filtered()
        self.stanceBegin_idx = self._get_stanceBegin_idx()
        num_steps = len(self.stanceBegin_idx)
        strideLength = np.zeros(num_steps)
        if num_steps == 0:
            return strideLength

        sample_init = 0
        #         sample_init = 10
        K = 0.23

        for i in range(len(self.stanceBegin_idx)):
            acc_max = np.max(acc_mag_filtered[int(sample_init):self.stanceBegin_idx[i]])
            acc_min = np.min(acc_mag_filtered[int(sample_init):self.stanceBegin_idx[i]])
            bounce = (acc_max - acc_min) ** (1 / 4)
            strideLength[i] = bounce * K * 2
            sample_init = self.stanceBegin_idx[i] + 1
        return strideLength

    def get_heading(self, ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw):
        acc_data = np.array([ax_raw, ay_raw, az_raw]).T
        gyr_data = np.array([gx_raw, gy_raw, gz_raw]).T * np.pi / 180

        # bias_gyr = np.mean(gyr_data[0:50*self.frequency,:], axis=0) # mean of the first 50 seconds
        bias_gyr = 0
        gyr_data = gyr_data - bias_gyr

        Q = np.tile(self.last_q, (len(gyr_data) + 1, 1))  # Allocate for quaternions
        for idx, (acc, gyr) in enumerate(
                zip(acc_data * self._G, gyr_data)):  # acc_data * self._G_G convert from g to m/s^2
            Q[idx + 1] = self.heading_estimator.updateIMU(Q[idx], acc=acc, gyr=gyr)
        self.Q = Q

        psi = []
        psi_flatten = []
        for idx, quatr in enumerate(Q):
            if idx in self.stanceBegin_idx:
                quatr = Quaternion(quatr).to_angles()
                psi_elem = quatr[2]
                psi.append(psi_elem)
                psi_flatten_elem = psi_elem * 180 / np.pi
                if psi_flatten_elem > 180 - 10 or psi_flatten_elem < -180 + 10:
                    psi_flatten_elem = 180
                elif 0 - 10 < psi_flatten_elem < 0 + 10:
                    psi_flatten_elem = 0
                elif 90 - 10 < psi_flatten_elem < 90 + 10:
                    psi_flatten_elem = 90
                elif -90 - 10 < psi_flatten_elem < -90 + 10:
                    psi_flatten_elem = -90
                psi_flatten_elem = psi_flatten_elem * np.pi / 180
                psi_flatten.append(psi_flatten_elem)
        psi = np.asarray(psi)
        psi_flatten = np.asarray(psi_flatten)
        return psi, psi_flatten

    def update(self, timestamp, ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw):
        self._set_acc_mag_filtered(ax_raw, ay_raw, az_raw)

        self.ts = np.r_[self.ts, timestamp]
        self.acc = np.r_[self.acc, list(zip(ax_raw, ay_raw, az_raw))]
        self.gyro = np.r_[self.gyro, list(zip(gx_raw, gy_raw, gz_raw))]

        self.strideLength = self.getStrideLength()
        if len(self.stanceBegin_idx) > 0:
            self.psi, self.psi_flatten = self.get_heading(self.acc[:, 0], self.acc[:, 1], self.acc[:, 2],
                                                          self.gyro[:, 0], self.gyro[:, 1], self.gyro[:, 2])
            last_step_idx = self.stanceBegin_idx[-1]
            self.ts = self.ts[last_step_idx + 1:]
            self.acc = self.acc[last_step_idx + 1:]
            self.gyro = self.gyro[last_step_idx + 1:]

            self.acc_mag_filtered = self.acc_mag_filtered[last_step_idx + 1:]

            self.last_acc_filt_binary = self.acc_filt_binary[last_step_idx]
            self.last_q = self.Q[last_step_idx]
        else:
            self.psi = np.empty(0)
            self.psi_flatten = np.empty(0)
