import numpy as np
import dttxml
from os import path

class Measurement(object):
    """
    This class is used to get measurement data from
    a dtt measurement. This class is used in conjuction with the
    ProcessSensingMeasurement and ProcessActuationMeasurement
    class to get the input for the MCMC and GPR analysis.

    This class uses the dttxml module to import transfer functions and ASDs.

    Parameters
    ----------
    meas_file: string
        String of the name of the xml file of the measurement.

    Returns
    -------

    Methods (Quick reference)
    -------
    get_raw_tf: Input two channels, get array of freq and TFs
    get_raw_asd: Input one channel, get array of freq and ASD
    get_set_of_channels: No input, get a set of the channels in meas_file.

    NOTES
    -----
    If the file is not given, the class will complain.
    """

    def __init__(self, meas_file):
        if path.exists(meas_file) is not True:
            raise IOError('The dtt file given does not exist.')

        # self.measurement = "sensing"

        # Data accessing object. It is called later.
        self.data_access = dttxml.DiagAccess(meas_file)

        # FIXME
        # For now, I can only access the averages and gps time from the
        # file only after I get the transfer function. There must be
        # a way to get them just after we access the file. For now,
        # I just initialize them here.
        self.averages = None
        self.gps_time = None

    def get_raw_tf(self, channelA, channelB, cohThresh=0):
        """
        Given two channels it returns the transfer function from the dtt
        file. Note that it gives channelB/channelA.
        In addition it populates averages and gps_time, calculates the
        uncertainty from coherence, and rejects data with coherence lower
        than the given threshold.

        Parameters
        ----------
        channelA: string
            Channel in the denominator of the TF
        channelB: string
            Channel in the numerator of the TF
        cohThresh: float
            Coherence threshold below which data from dtt are rejected

        Returns
        -------
        Returns a comlex array of (Nx4), where N is the number of data
        points, and the four columns are:
        Column 1: frequencies (frequencies of the data in Hz)
        Column 2: tf (transfer function of B/A)
        Column 3: coh (coherence of B/A measurement)
        Column 4: unc (the uncertainty from coherence)
        """
        if (channelA in self.get_set_of_channels()) and\
           (channelB in self.get_set_of_channels()):
            # Transfer function data holder. xfer is a method that comes from
            # dttxml. xfer takes channelB first and channelA second, which
            # is the opposite order that I am using. Whatever the ordering,
            # channelA always means the denominator and channelB is the
            # numerator.This should work for both Swept Sine
            # or FFT (aka broadband) measurements.
            tf_holder = self.data_access.xfer(channelB, channelA)
            self.averages = tf_holder.averages
            self.gps_time = tf_holder.gps_second
            # Frequency of the measurement in Hz
            frequencies = np.array(tf_holder.FHz)
            # Transfer function of B/A, complex array
            tf = np.array(tf_holder.xfer)
            # Coherence function of B/A, real array
            coh = np.array(tf_holder.coh)
            # Uncertainty from coherence
            unc = np.sqrt(1.0-coh) / \
                (2.0*(coh+1e-6)*self.averages)
            # Find good coherence and intersecting points
            frequencies = frequencies[(coh > cohThresh)]
            tf = tf[(coh > cohThresh)]
            unc = unc[(coh > cohThresh)]
            coh = coh[(coh > cohThresh)]

            return frequencies, tf, coh, unc
        else:
            raise ValueError('Invalid channel for transfer function.')

    def get_raw_asd(self, channelA):
        """
        It returns the ASD of channelA from the dtt file.
        This will only work for a FFT (aka broadband) type measurement.

        Parameters
        ----------
        channelA: string
            ASD channel.

        Returns
        -------
        Returns two float array:
        Array 1: frequencies (frequencies of the data in Hz)
        Array 2: asd (ASD of channelA)
        """

        if (channelA in self.get_set_of_channels()):
            # This will only work for a FFT (aka broadband) type measurement
            asd_holder = self.data_access.asd(channelA)
            self.averages = asd_holder.averages
            self.gps_time = asd_holder.gps_second
            # Frequency of the measurement in Hz
            frequencies = np.array(asd_holder.FHz)
            # Amplitude Spectral Density of requested channel
            asd = np.array(asd_holder.asd)
            return frequencies, asd
        else:
            raise ValueError('Invalid channel for asd.')

    def get_set_of_channels(self):
        """
        Method to get the channels in the measurement file.
        Annoyingly, the channels are given in a set.
        Parameters
        ----------

        Returns
        -------
        channels: A python set with all the channels in the measurement file
        of this object.
        """
        channels = self.data_access.channels()
        return channels[0]
