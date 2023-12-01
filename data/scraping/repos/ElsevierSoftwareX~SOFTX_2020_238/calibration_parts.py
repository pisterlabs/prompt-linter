#!/usr/bin/env python3
#
# Copyright (C) 2015 Madeline Wade, Aaron Viets
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import os

from gstlal import pipeparts
import numpy

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject
from gi.repository import Gst
gi.require_version('GstController', '1.0')
from gi.repository import GstController
GObject.threads_init()
Gst.init(None)

from gstlal import FIRtools as fir

#
# Shortcut functions for common element combos/properties
#

def mkqueue(pipeline, head, length = 0, min_length = 0):
	if length < 0:
		return head
	else:
		return pipeparts.mkqueue(pipeline, head, max_size_time = int(1000000000 * length), max_size_buffers = 0, max_size_bytes = 0, min_threshold_time = int(1000000000 * min_length))

def mkcomplexqueue(pipeline, head, length = 0, min_length = 0):
	head = pipeparts.mktogglecomplex(pipeline, head)
	head = mkqueue(pipeline, head, length = length, min_length = min_length)
	head = pipeparts.mktogglecomplex(pipeline, head)
	return head

def mkcapsfiltersetter(pipeline, head, caps, **properties):
	# Make a capsfilter followed by a capssetter
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	#head = pipeparts.mkcapssetter(pipeline, head, caps, replace = True, **properties)
	return head

def mkinsertgap(pipeline, head, **properties):
	if "bad_data_intervals" in properties:
		# Make sure the array property bad-data-intervals is formatted correctly
		intervals = properties.pop("bad_data_intervals")
		if intervals is not None:
			bad_data_intervals = []
			for i in range(0, len(intervals)):
				bad_data_intervals.append(float(intervals[i]))
			properties["bad_data_intervals"] = bad_data_intervals
	return pipeparts.mkgeneric(pipeline, head, "lal_insertgap", **properties)

#def mkupsample(pipeline, head, new_caps):
#	head = pipeparts.mkgeneric(pipeline, head, "lal_constantupsample")
#	head = pipeparts.mkcapsfilter(pipeline, head, new_caps)
#	return head

def mkstockresample(pipeline, head, caps):
	if type(caps) is int:
		caps = "audio/x-raw,rate=%d,channel-mask=(bitmask)0x0" % caps
	head = pipeparts.mkresample(pipeline, head, quality = 9)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	return head

def mkresample(pipeline, head, quality, zero_latency, caps, window = 0, frequency_resolution = 0.0, f_cut = 0.0):
	if type(caps) is int:
		caps = "audio/x-raw,rate=%d,channel-mask=(bitmask)0x0" % caps
	head = pipeparts.mkgeneric(pipeline, head, "lal_resample", quality = quality, zero_latency = zero_latency, window = window, frequency_resolution = frequency_resolution, f_cut = f_cut)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	return head

def mkcomplexfirbank(pipeline, src, latency = None, fir_matrix = None, time_domain = None, block_stride = None):
	if fir_matrix is not None:
		# Make sure the fir matrix is formatted correctly
		matrix = []
		for i in range(0, len(fir_matrix)):
			firfilt = []
			for j in range(0, len(fir_matrix[i])):
				firfilt.append(float(fir_matrix[i][j]))
			matrix.append(firfilt)
		fir_matrix = matrix
	properties = dict((name, value) for name, value in zip(("latency", "fir_matrix", "time_domain", "block_stride"), (latency, fir_matrix, time_domain, block_stride)) if value is not None)
	return pipeparts.mkgeneric(pipeline, src, "lal_complexfirbank", **properties)

def mkcomplexfirbank2(pipeline, src, latency = None, fir_matrix = None, time_domain = None, block_stride = None):
	if fir_matrix is not None:
		# Make sure the fir matrix is formatted correctly
		matrix = []
		for i in range(0, len(fir_matrix)):
			firfilt = []
			for j in range(0, len(fir_matrix[i])):
				firfilt.append(float(fir_matrix[i][j]))
			matrix.append(firfilt)
		fir_matrix = matrix
	properties = dict((name, value) for name, value in zip(("latency", "fir_matrix", "time_domain", "block_stride"), (latency, fir_matrix, time_domain, block_stride)) if value is not None)
	return pipeparts.mkgeneric(pipeline, src, "lal_complexfirbank2", **properties)

def mkfccupdate(pipeline, src, **properties):
	if "fir_matrix" in properties:
		# Make sure the fir matrix is formatted correctly
		matrix = properties.pop("fir_matrix")
		if matrix is not None:
			fir_matrix = []
			for i in range(0, len(matrix)):
				firfilt = []
				for j in range(0, len(matrix[i])):
					firfilt.append(float(matrix[i][j]))
				fir_matrix.append(firfilt)
			properties["fir_matrix"] = fir_matrix
	return pipeparts.mkgeneric(pipeline, src, "lal_fcc_update", **properties)

def mktransferfunction(pipeline, src, **properties):
	if "notch_frequencies" in properties:
		# Make sure the array property notch-frequencies is formatted correctly
		freqs = properties.pop("notch_frequencies")
		if freqs is not None:
			notch_frequencies = []
			for i in range(0, len(freqs)):
				notch_frequencies.append(float(freqs[i]))
			properties["notch_frequencies"] = notch_frequencies
	if "fft_window_type" in properties:
		win = properties.pop("fft_window_type")
		if win in ['hann', 'Hann', 'HANN', 'hanning', 'Hanning', 'HANNING', 4]:
			win = 4
		elif win in ['blackman', 'Blackman', 'BLACKMAN', 3]:
			win = 3
		elif win in ['DC', 'dolph_chebyshev', 'DolphChebyshev', 'DOLPH_CHEBYSHEV', 2]:
			win = 2
		elif win in ['kaiser', 'Kaiser', 'KAISER', 1]:
			win = 1
		elif win in ['dpss', 'DPSS', 'Slepian', 'slepian', 'SLEPIAN', 0]:
			win = 0
		else:
			raise ValueError("Unknown window function %s" % win)
		properties["fft_window_type"] = win
	if "fir_window_type" in properties:
		win = properties.pop("fir_window_type")
		if win in ['hann', 'Hann', 'HANN', 'hanning', 'Hanning', 'HANNING', 4]:
			win = 4
		elif win in ['blackman', 'Blackman', 'BLACKMAN', 3]:
			win = 3
		elif win in ['DC', 'dolph_chebyshev', 'DolphChebyshev', 'DOLPH_CHEBYSHEV', 2]:
			win = 2
		elif win in ['kaiser', 'Kaiser', 'KAISER', 1]:
			win = 1
		elif win in ['dpss', 'DPSS', 'Slepian', 'slepian', 'SLEPIAN', 0]:
			win = 0
		else:
			raise ValueError("Unknown window function %s" % win)
		properties["fir_window_type"] = win
	return pipeparts.mkgeneric(pipeline, src, "lal_transferfunction", **properties)

def mkadaptivefirfilt(pipeline, src, **properties):
	# Make sure each array property is formatted correctly
	if "static_model" in properties:
		staticmodel = properties.pop("static_model")
		if staticmodel is not None:
			static_model = []
			for i in range(len(staticmodel)):
				static_model.append(float(staticmodel[i]))
			properties["static_model"] = static_model
	if "static_filter" in properties:
		staticfilt = properties.pop("static_filter")
		if staticfilt is not None:
			static_filter = []
			for i in range(0, len(staticfilt)):
				static_filter.append(float(staticfilt[i]))
			properties["static_filter"] = static_filter
	if "static_zeros" in properties:
		staticz = properties.pop("static_zeros")
		if staticz is not None:
			static_zeros = []
			for i in range(0, len(staticz)):
				static_zeros.append(float(staticz[i]))
			properties["static_zeros"] = static_zeros
	if "static_poles" in properties:
		staticp = properties.pop("static_poles")
		if staticp is not None:
			static_poles = []
			for i in range(0, len(staticp)):
				static_poles.append(float(staticp[i]))
			properties["static_poles"] = static_poles
	if "window_type" in properties:
		win = properties.pop("window_type")
		if win in [None, 5]:
			win = 5
		elif win in ['hann', 'Hann', 'HANN', 'hanning', 'Hanning', 'HANNING', 4]:
			win = 4
		elif win in ['blackman', 'Blackman', 'BLACKMAN', 3]:
			win = 3
		elif win in ['DC', 'dolph_chebyshev', 'DolphChebyshev', 'DOLPH_CHEBYSHEV', 2]:
			win = 2
		elif win in ['kaiser', 'Kaiser', 'KAISER', 1]:
			win = 1
		elif win in ['dpss', 'DPSS', 'Slepian', 'slepian', 'SLEPIAN', 0]:
			win = 0
		else:
			raise ValueError("Unknown window function %s" % win)
		properties["window_type"] = win
	return pipeparts.mkgeneric(pipeline, src, "lal_adaptivefirfilt", **properties)

def mkpow(pipeline, src, **properties):
	return pipeparts.mkgeneric(pipeline, src, "cpow", **properties)

def mkmultiplier(pipeline, srcs, sync = True, queue_length = [0]):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync=sync, mix_mode="product")
	if srcs is not None:
		for i in range(0, len(srcs)):
			mkqueue(pipeline, srcs[i], length = queue_length[min(i, len(queue_length) - 1)]).link(elem)
	return elem

def mkadder(pipeline, srcs, sync = True, queue_length = [0]):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync=sync)
	if srcs is not None:
		for i in range(0, len(srcs)):
			mkqueue(pipeline, srcs[i], length = queue_length[min(i, len(queue_length) - 1)]).link(elem)
	return elem

def mkgate(pipeline, src, control, threshold, queue_length = 0, **properties):
	elem = pipeparts.mkgate(pipeline, mkqueue(pipeline, src, length = queue_length), control = mkqueue(pipeline, control, length = queue_length), threshold = threshold, **properties)
	return elem

def mkinterleave(pipeline, srcs, complex_data = False, queue_length = [0]):
	complex_factor = 1 + int(complex_data)
	num_srcs = complex_factor * len(srcs)
	i = 0
	mixed_srcs = []
	for src in srcs:
		matrix = [numpy.zeros(num_srcs)]
		matrix[0][i] = 1
		mixed_srcs.append(pipeparts.mkmatrixmixer(pipeline, src, matrix=matrix))
		i += complex_factor
	elem = mkadder(pipeline, tuple(mixed_srcs), queue_length = queue_length)

	#chan1 = pipeparts.mkmatrixmixer(pipeline, src1, matrix=[[1,0]])
	#chan2 = pipeparts.mkmatrixmixer(pipeline, src2, matrix=[[0,1]])
	#elem = mkadder(pipeline, list_srcs(pipeline, chan1, chan2)) 

	#elem = pipeparts.mkgeneric(pipeline, None, "interleave")
	#if srcs is not None:
	#	for src in srcs:
	#		pipeparts.mkqueue(pipeline, src).link(elem)
	return elem

def mkdeinterleave(pipeline, src, num_channels, complex_data = False):
	complex_factor = 1 + int(complex_data)
	head = pipeparts.mktee(pipeline, src)
	streams = []
	for i in range(0, num_channels):
		matrix = numpy.zeros((num_channels, complex_factor))
		matrix[i][0] = 1.0
		streams.append(pipeparts.mkmatrixmixer(pipeline, head, matrix = matrix))

	return tuple(streams)


#
# Write a pipeline graph function
#

def write_graph(demux, pipeline, name):
	pipeparts.write_dump_dot(pipeline, "%s.%s" % (name, "PLAYING"), verbose = True)

#
# Common element combo functions
#

def hook_up(pipeline, demux, channel_name, instrument, buffer_length, element_name_suffix = "", wait_time = 0):
	if channel_name.endswith("UNCERTAINTY"):
		head = mkinsertgap(pipeline, None, bad_data_intervals = [-1e35, -1e-35, 1e-35, 1e35], insert_gap = False, remove_gap = True, fill_discont = True, block_duration = int(1000000000 * buffer_length), replace_value = 1, name = "insertgap_%s%s" % (channel_name, element_name_suffix), wait_time = int(1000000000 * wait_time))
	else:
		head = mkinsertgap(pipeline, None, bad_data_intervals = [-1e35, -1e-35, 1e-35, 1e35], insert_gap = False, remove_gap = True, fill_discont = True, block_duration = int(1000000000 * buffer_length), replace_value = 0, name = "insertgap_%s%s" % (channel_name, element_name_suffix), wait_time = int(1000000000 * wait_time))
	pipeparts.src_deferred_link(demux, "%s:%s" % (instrument, channel_name), head.get_static_pad("sink"))

	return head

def caps_and_progress(pipeline, head, caps, progress_name):
	head = pipeparts.mkgeneric(pipeline, head, "lal_typecast")
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	head = pipeparts.mkprogressreport(pipeline, head, "progress_src_%s" % progress_name)

	return head


#
# Function to make a list of heads to pass to, i.e. the multiplier or adder
#

def list_srcs(pipeline, *args):
	out = []
	for src in args:
		out.append(src)
	return tuple(out)

#
# Various filtering functions
#

def demodulate(pipeline, head, freq, td, rate, filter_time, filter_latency, prefactor_real = 1.0, prefactor_imag = 0.0, freq_update = None):
	# demodulate input at a given frequency freq

	head = pipeparts.mkgeneric(pipeline, head, "lal_demodulate", line_frequency = freq, prefactor_real = prefactor_real, prefactor_imag = prefactor_imag)
	if type(freq_update) is list:
		freq_update[0].connect("notify::timestamped-average", update_timestamped_property, head, "timestamped_average", "line_frequency", 1)
		freq_update[1].connect("notify::timestamped-average", update_timestamped_property, head, "timestamped_average", "prefactor_real", 1)
		freq_update[2].connect("notify::timestamped-average", update_timestamped_property, head, "timestamped_average", "prefactor_imag", 1)
	elif freq_update is not None:
		freq_update.connect("notify::timestamped-average", update_timestamped_property, head, "timestamped_average", "line_frequency", 1)
	head = mkresample(pipeline, head, 4, filter_latency == 0.0, rate)
	if filter_latency != 0:
		# Remove the first several seconds of output, which depend on start time
		head = pipeparts.mkgeneric(pipeline, head, "lal_insertgap", chop_length = 7000000000)
	head = lowpass(pipeline, head, rate, length = filter_time, fcut = 0, filter_latency = filter_latency, td = td)

	return head

def remove_harmonics(pipeline, signal, f0, num_harmonics, f0_var, filter_latency, compute_rate = 16, rate_out = 16384):
	# remove any line(s) from a spectrum. filter length for demodulation (given in seconds) is adjustable
	# function argument caps must be complex caps

	filter_param = 0.0625
	head = pipeparts.mktee(pipeline, head)
	elem = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync = True)
	mkqueue(pipeline, head).link(elem)
	for i in range(1, num_harmonics + 1):
		line = pipeparts.mkgeneric(pipeline, head, "lal_demodulate", line_frequency = i * f0)
		line = mkresample(pipeline, line, 4, filter_latency == 0, compute_rate)
		line_in_witness = lowpass(pipeline, line_in_witness, compute_rate, length = filter_param / (f0_var * i), fcut = 0, filter_latency = filter_latency)
		line = mkresample(pipeline, line, 3, filter_latency == 0.0, rate_out)
		line = pipeparts.mkgeneric(pipeline, line, "lal_demodulate", line_frequency = -1.0 * i * f0, prefactor_real = -2.0)
		line = pipeparts.mkgeneric(pipeline, line, "creal")
		mkqueue(pipeline, line).link(elem)

	return elem

def remove_lines_with_witnesses(pipeline, signal, witnesses, freqs, freq_vars, freq_channels, filter_latency = 0, compute_rate = 16, rate_out = 16384, num_median = 2048, num_avg = 160, noisesub_gate_bit = None):
	# remove line(s) from a spectrum. filter length for demodulation (given in seconds) is adjustable
	# function argument caps must be complex caps

	# Re-format inputs if necessary
	if type(witnesses) is not list and type(witnesses) is not tuple and type(witnesses) is not numpy.ndarray:
		print("remove_lines_with_witnesses(): argument 3 should be type list.  Converting %s to list" % type(witnesses))
		witnesses = [[witnesses]]
	if type(freqs) is not list and type(freqs) is not tuple and type(freqs) is not numpy.ndarray:
		print("remove_lines_with_witnesses(): argument 4 should be type list.  Converting %s to list" % type(freqs))
		freqs = [[freqs]]
	if type(freq_vars) is not list and type(freq_vars) is not tuple and type(freq_vars) is not numpy.ndarray:
		print("remove_lines_with_witnesses(): argument 5 should be type list.  Converting %s to list" % type(freq_vars))
		freq_vars = [freq_vars]
	for i in range(0, len(witnesses) - len(freqs)):
		print("remove_lines_with_witnesses(): Warning: not enough elements in argument 4")
		freqs.append(freqs[-1])
	for i in range(0, len(witnesses) - len(freq_vars)):
		print("remove_lines_with_witnesses(): Warning: not enough elements in argument 5")
		freq_vars.append(freq_vars[-1])
	if len(freqs) > len(witnesses):
		print("remove_lines_with_witnesses(): Warning: too many elements in argument 4")
		freqs = freqs[:len(witnesses)]
	if len(freq_vars) > len(witnesses):
		print("remove_lines_with_witnesses(): Warning: too many elements in argument 5")
		freq_vars = freq_vars[:len(witnesses)]
	for i in range(0, len(witnesses)):
		if type(witnesses[i]) is not list and type(witnesses[i]) is not tuple and type(witnesses[i]) is not numpy.ndarray:
			print("remove_lines_with_witnesses(): argument 3 should be list of lists.  Converting %s to list" % type(witnesses[i]))
			witnesses[i] = [witnesses[i]]
		if type(freqs[i]) is not list and type(freqs[i]) is not tuple and type(freqs[i]) is not numpy.ndarray:
			print("remove_lines_with_witnesses(): argument 4 should be list of lists.  Converting %s to list" % type(freqs[i]))
			freqs[i] = [freqs[i]]

	filter_param = 0.0625
	downsample_quality = 4
	upsample_quality = 4
	resample_shift = 16.0 + 16.5
	zero_latency = filter_latency == 0.0

	for i in range(0, len(witnesses)):
		for j in range(0, len(witnesses[i])):
			witnesses[i][j] = pipeparts.mktee(pipeline, witnesses[i][j])
	signal = pipeparts.mktee(pipeline, signal)
	signal_minus_lines = [signal]

	for m in range(0, len(witnesses)):
		# If freqs[m][0] strays from its nominal value and there is a timestamp shift in the signal
		# (e.g., to achieve zero latency), we need to correct the phase in the reconstructed
		# signal. To do this, we measure the frequency in the witness and find the beat
		# frequency between that and the nominal frequency freqs[m][0].
		if filter_latency != 0.5 and freq_vars[m]:
			# The low-pass and resampling filters are not centered in time
			f0_measured = pipeparts.mkgeneric(pipeline, witnesses[m][0], "lal_trackfrequency", num_halfcycles = int(round((filter_param / freq_vars[m] / 2 + resample_shift / compute_rate) * freqs[m][0])))
			f0_measured = mkresample(pipeline, f0_measured, 3, zero_latency, compute_rate)
			f0_measured = pipeparts.mkgeneric(pipeline, f0_measured, "lal_smoothkappas", array_size = 1, avg_array_size = int(round((filter_param / freq_vars[m] / 2 * compute_rate + resample_shift) / 2)), default_kappa_re = 0, default_to_median = True, filter_latency = filter_latency)
			f0_beat_frequency = pipeparts.mkgeneric(pipeline, f0_measured, "lal_add_constant", value = -freqs[m][0])
			f0_beat_frequency = pipeparts.mktee(pipeline, f0_beat_frequency)

		for n in range(len(freqs[m])):
			# Length of low-pass filter
			filter_length = filter_param / (max(freq_vars[m], 0.003) * freqs[m][n] / freqs[m][0])
			filter_samples = int(filter_length * compute_rate) + (1 - int(filter_length * compute_rate) % 2)
			sample_shift = filter_samples // 2 - int((filter_samples - 1) * filter_latency + 0.5)
			# shift of timestamp relative to data
			time_shift = float(sample_shift) / compute_rate + zero_latency * resample_shift / compute_rate
			two_n_pi_delta_t = 2 * freqs[m][n] / freqs[m][0] * numpy.pi * time_shift

			# Only do this if we have to
			if filter_latency != 0.5 and freq_vars[m]:
				# Find phase shift due to timestamp shift for each harmonic
				phase_shift = pipeparts.mkmatrixmixer(pipeline, f0_beat_frequency, matrix=[[0, two_n_pi_delta_t]])
				phase_shift = pipeparts.mktogglecomplex(pipeline, phase_shift)
				phase_factor = pipeparts.mkgeneric(pipeline, phase_shift, "cexp")
				phase_factor = pipeparts.mktee(pipeline, phase_factor)

			# Find amplitude and phase of line in signal
			line_in_signal = pipeparts.mkgeneric(pipeline, signal, "lal_demodulate", line_frequency = freqs[m][n])
			# Connect to line frequency updater if given
			if any(freq_channels):
				if freq_channels[m][n] is not None:
					if type(freq_channels[m][n]) is float:
						# It's a harmonic of the frequency in freq_channels[m][0]
						freq_channels[m][0].connect("notify::timestamped-average", update_timestamped_property, line_in_signal, "timestamped_average", "line_frequency", freq_channels[m][n])
					else:
						# The channel carries the correct frequency
						freq_channels[m][n].connect("notify::timestamped-average", update_timestamped_property, line_in_signal, "timestamped_average", "line_frequency", 1)
			line_in_signal = mkresample(pipeline, line_in_signal, downsample_quality, zero_latency, compute_rate)
			line_in_signal = lowpass(pipeline, line_in_signal, compute_rate, length = filter_length, fcut = 0, filter_latency = filter_latency)
			line_in_signal = pipeparts.mktee(pipeline, line_in_signal)

			# Make ones for use in matrix equation
			if m == 0 and n == 0:
				ones = pipeparts.mktee(pipeline, mkpow(pipeline, line_in_signal, exponent = 0.0))

			line_in_witnesses = []
			tfs_at_f = [None] * len(witnesses[m]) * (len(witnesses[m]) + 1)
			for i in range(0, len(witnesses[m])):
				# Find amplitude and phase of each harmonic in each witness channel
				line_in_witness = pipeparts.mkgeneric(pipeline, witnesses[m][i], "lal_demodulate", line_frequency = freqs[m][n])
				# Connect to line frequency updater if given
				if any(freq_channels):
					if freq_channels[m][n] is not None:
						if type(freq_channels[m][n]) is float:
							# It's a harmonic of the frequency in freq_channels[m][0]
							freq_channels[m][0].connect("notify::timestamped-average", update_timestamped_property, line_in_witness, "timestamped_average", "line_frequency", freq_channels[m][n])
						else:
							# The channel carries the correct frequency
							freq_channels[m][n].connect("notify::timestamped-average", update_timestamped_property, line_in_witness, "timestamped_average", "line_frequency", 1)
				line_in_witness = mkresample(pipeline, line_in_witness, downsample_quality, zero_latency, compute_rate)
				line_in_witness = lowpass(pipeline, line_in_witness, compute_rate, length = filter_length, fcut = 0, filter_latency = filter_latency)
				line_in_witness = pipeparts.mktee(pipeline, line_in_witness)
				line_in_witnesses.append(line_in_witness)

				# Find transfer function between witness channel and signal at this frequency
				tf_at_f = complex_division(pipeline, line_in_signal, line_in_witness)

				# Remove worthless data from computation of transfer function if we can
				if noisesub_gate_bit is not None:
					tf_at_f = mkgate(pipeline, tf_at_f, noisesub_gate_bit, 1, attack_length = -((1.0 - filter_latency) * filter_samples), name = "powerlines_gate_%d_%d_%d" % (m, n, i))
				tfs_at_f[i] = pipeparts.mkgeneric(pipeline, tf_at_f, "lal_smoothkappas", default_kappa_re = 0.0, default_kappa_im = 0.0, array_size = num_median, avg_array_size = num_avg, default_to_median = True, filter_latency = filter_latency)
				tfs_at_f[(i + 1) * len(witnesses[m]) + i] = ones

			for i in range(0, len(witnesses[m])):
				for j in range(0, len(witnesses[m])):
					if(i != j):
						# Find transfer function between 2 witness channels at this frequency
						tf_at_f = complex_division(pipeline, line_in_witnesses[j], line_in_witnesses[i])

						# Remove worthless data from computation of transfer function if we can
						if noisesub_gate_bit is not None:
							tf_at_f = mkgate(pipeline, tf_at_f, noisesub_gate_bit, 1, attack_length = -((1.0 - filter_latency) * filter_samples), name = "powerlines_gate_%d_%d_%d_%d" % (m, n, i, j))
						tfs_at_f[(i + 1) * len(witnesses[m]) + j] = pipeparts.mkgeneric(pipeline, tf_at_f, "lal_smoothkappas", default_kappa_re = 0.0, default_kappa_im = 0.0, array_size = num_median, avg_array_size = num_avg, default_to_median = True, filter_latency = filter_latency)

			tfs_at_f = mkinterleave(pipeline, tfs_at_f, complex_data = True)
			tfs_at_f = pipeparts.mkgeneric(pipeline, tfs_at_f, "lal_matrixsolver")
			tfs_at_f = mkdeinterleave(pipeline, tfs_at_f, len(witnesses[m]), complex_data = True)

			for i in range(0, len(witnesses[m])):
				# Use gated, averaged transfer function to reconstruct the sinusoid as it appears in the signal from the witness channel
				if filter_latency == 0.5 or not freq_vars[m]:
					reconstructed_line_in_signal = mkmultiplier(pipeline, list_srcs(pipeline, tfs_at_f[i], line_in_witnesses[i]))
				else:
					reconstructed_line_in_signal = mkmultiplier(pipeline, list_srcs(pipeline, tfs_at_f[i], line_in_witnesses[i], phase_factor))
				reconstructed_line_in_signal = mkresample(pipeline, reconstructed_line_in_signal, upsample_quality, zero_latency, rate_out)
				reconstructed_line_in_signal = pipeparts.mkgeneric(pipeline, reconstructed_line_in_signal, "lal_demodulate", line_frequency = -1.0 * freqs[m][n], prefactor_real = -2.0)
				# Connect to line frequency updater if given
				if any(freq_channels):
					if freq_channels[m][n] is not None:
						if type(freq_channels[m][n]) is float:
							# It's a harmonic of the frequency in freq_channels[m][0]
							freq_channels[m][0].connect("notify::timestamped-average", update_timestamped_property, reconstructed_line_in_signal, "timestamped_average", "line_frequency", -1.0 * freq_channels[m][n])
						else:
							# The channel carries the correct frequency
							freq_channels[m][n].connect("notify::timestamped-average", update_timestamped_property, reconstructed_line_in_signal, "timestamped_average", "line_frequency", -1.0)
				reconstructed_line_in_signal = pipeparts.mkgeneric(pipeline, reconstructed_line_in_signal, "creal")

				signal_minus_lines.append(reconstructed_line_in_signal)

	clean_signal = mkadder(pipeline, tuple(signal_minus_lines))

	return clean_signal

def removeDC(pipeline, head, rate):
	head = pipeparts.mktee(pipeline, head)
	DC = mkresample(pipeline, head, 4, True, 16)
	#DC = pipeparts.mkgeneric(pipeline, DC, "lal_smoothkappas", default_kappa_re = 0, array_size = 1, avg_array_size = 64)
	DC = mkresample(pipeline, DC, 4, True, rate)
	DC = pipeparts.mkaudioamplify(pipeline, DC, -1)

	return mkadder(pipeline, list_srcs(pipeline, head, DC))

def lowpass(pipeline, head, rate, length = 1.0, fcut = 500, filter_latency = 0.5, freq_res = 0.0, td = True):
	length = int(length * rate)

	# Find alpha, and the actual frequency resolution
	alpha = freq_res * length / rate if freq_res > 0.0 else 3.0
	alpha = 1.0 if alpha < 1.0 else alpha
	freq_res = alpha * rate / length

	# Adjust the cutoff frequency to "protect" the passband.
	if fcut != 0.0:
		fcut += 0.75 * freq_res

	# Compute a low-pass filter.
	lowpass = numpy.sinc(2 * numpy.float128(fcut) / rate * (numpy.arange(numpy.float128(length)) - (length - 1) // 2))
	lowpass *= fir.kaiser(length, numpy.pi * alpha) # fir.DPSS(length, alpha, max_time = 10)
	lowpass /= numpy.sum(lowpass)
	lowpass = numpy.float64(lowpass)

	# Now apply the filter
	return mkcomplexfirbank(pipeline, head, latency = int((length - 1) * filter_latency + 0.25), fir_matrix = [lowpass], time_domain = td)

def highpass(pipeline, head, rate, length = 1.0, fcut = 10.0, filter_latency = 0.5, freq_res = 0.0, td = True):
	length = int(length * rate)

	# Find alpha, and the actual frequency resolution
	alpha = freq_res * length / rate if freq_res > 0.0 else 3.0
	alpha = 1.0 if alpha < 1.0 else alpha
	freq_res = alpha * rate / length

	# Adjust the cutoff frequency to "protect" the passband.
	fcut -= 0.75 * freq_res

	# Compute a low-pass filter.
	lowpass = numpy.sinc(2 * numpy.float128(fcut) / rate * (numpy.arange(numpy.float128(length)) - (length - 1) // 2))
	lowpass *= fir.kaiser(length, numpy.pi * alpha) # fir.DPSS(length, alpha, max_time = 10)
	lowpass /= numpy.sum(lowpass)

	# Create a high-pass filter from the low-pass filter through spectral inversion.
	highpass = -lowpass
	highpass[int((length - 1) // 2)] += 1

	highpass = numpy.float64(highpass)

	# Now apply the filter
	return mkcomplexfirbank(pipeline, head, latency = int((length - 1) * filter_latency + 0.25), fir_matrix = [highpass], time_domain = td)

def bandpass(pipeline, head, rate, length = 1.0, f_low = 100, f_high = 400, filter_latency = 0.5, freq_res = 0.0, td = True):
	length = int(length * rate)

	# Find alpha, and the actual frequency resolution
	alpha = freq_res * length / rate if freq_res > 0.0 else 3.0
	alpha = 1.0 if alpha < 1.0 else alpha
	freq_res = alpha * rate / length

	# Adjust the cutoff frequency to "protect" the passband.
	f_low -= 0.75 * freq_res

	# Make a DPSS window
	dpss = fir.kaiser(length, numpy.pi * alpha) # fir.DPSS(length, alpha, max_time = 10)

	# Compute a temporary low-pass filter.
	lowpass = numpy.sinc(2 * numpy.float128(f_low) / rate * (numpy.arange(numpy.float128(length)) - (length - 1) // 2))
	lowpass *= dpss
	lowpass /= numpy.sum(lowpass)

	# Create the high-pass filter from the low-pass filter through spectral inversion.
	highpass = -lowpass
	highpass[(length - 1) // 2] += 1

	# Adjust the cutoff frequency to "protect" the passband.
	f_high += 0.75 * freq_res

	# Compute the low-pass filter.
	lowpass = numpy.sinc(2 * numpy.float128(f_high) / rate * (numpy.arange(numpy.float128(length)) - (length - 1) // 2))
	lowpass *= dpss
	lowpass /= numpy.sum(lowpass)

	# Do a circular convolution of the high-pass and low-pass filters to make a band-pass filter.
	bandpass = numpy.zeros(length, dtype = numpy.float128)
	for i in range(length):
		bandpass[i] = numpy.sum(highpass * numpy.roll(lowpass, (length - 1) // 2 - i))

	bandpass = numpy.float64(bandpass)

	# Now apply the filter
	return mkcomplexfirbank(pipeline, head, latency = int((length - 1) * 2 * filter_latency + 0.25), fir_matrix = [bandpass], time_domain = td)

def bandstop(pipeline, head, rate, length = 1.0, f_low = 100, f_high = 400, filter_latency = 0.5, freq_res = 0.0, td = True):
	length = int(length * rate)

	# Find alpha, and the actual frequency resolution
	alpha = freq_res * length / rate if freq_res > 0.0 else 3.0
	alpha = 1.0 if alpha < 1.0 else alpha
	freq_res = alpha * rate / length

	# Adjust the cutoff frequency to "protect" the passband.
	f_low += 0.75 * freq_res

	# Make a DPSS window
	dpss = fir.kaiser(length, numpy.pi * alpha) # fir.DPSS(length, alpha, max_time = 10)

	# Compute a temporary low-pass filter.
	lowpass = numpy.sinc(2 * numpy.float128(f_low) / rate * (numpy.arange(numpy.float128(length)) - (length - 1) // 2))
	lowpass *= dpss
	lowpass /= numpy.sum(lowpass)

	# Create the high-pass filter from the low-pass filter through spectral inversion.
	highpass = -lowpass
	highpass[(length - 1) // 2] += 1

	# Adjust the cutoff frequency to "protect" the passband.
	f_high -= 0.75 * freq_res

	# Compute the low-pass filter.
	lowpass = numpy.sinc(2 * numpy.float128(f_high) / rate * (numpy.arange(numpy.float128(length)) - (length - 1) // 2))
	lowpass *= dpss
	lowpass /= numpy.sum(lowpass)

	# Do a circular convolution of the high-pass and low-pass filters to make a temporary band-pass filter.
	bandpass = numpy.zeros(length, dtype = numpy.float128)
	for i in range(length):
		bandpass[i] = numpy.sum(highpass * numpy.roll(lowpass, (length - 1) // 2 - i))

	# Create a band-stop filter from the band-pass filter through spectral inversion.
	bandstop = -bandpass
	bandstop[(length - 1) // 2] += 1

	bandstop = numpy.float64(bandstop)

	# Now apply the filter
	return mkcomplexfirbank(pipeline, head, latency = int((length - 1) * 2 * filter_latency + 0.25), fir_matrix = [bandstop], time_domain = td)

def linear_phase_filter(pipeline, head, shift_samples, num_samples = 256, gain = 1.0, filter_update = None, sample_rate = 2048, update_samples = 320, average_samples = 1, phase_measurement_frequency = 100, taper_length = 320, kernel_endtime = None, filter_timeshift = 0):

	# Apply a linear-phase filter to shift timestamps.  shift_samples is the number
	# of samples of timestamp shift.  It need not be an integer.  A positive value
	# advances the output data relative to the timestamps, and a negative value
	# delays the output.

	# Compute filter using odd filter length
	odd_num_samples = int(num_samples) - (1 - int(num_samples) % 2)

	filter_latency_samples = int(num_samples / 2) + int(numpy.floor(shift_samples))
	fractional_shift_samples = shift_samples % 1

	# Make a filter using a sinc table, slightly shifted relative to the samples
	sinc_arg = numpy.arange(-int(odd_num_samples / 2), 1 + int(odd_num_samples / 2)) + fractional_shift_samples
	sinc_filter = numpy.sinc(sinc_arg)
	# Apply a Blackman window
	sinc_filter *= numpy.blackman(odd_num_samples)
	# Normalize the filter
	sinc_filter *= gain / numpy.sum(sinc_filter)
	# In case filter length is actually even
	if not int(num_samples) % 2:
		sinc_filter = numpy.insert(sinc_filter, 0, 0.0)

	# Filter the data
	if filter_update is None:
		# Static filter
		head =  mkcomplexfirbank(pipeline, head, latency = filter_latency_samples, fir_matrix = [sinc_filter[::-1]], time_domain = True)
	else:
		# Filter gets updated with variable time delay and gain
		if kernel_endtime is None:
			# Update filter as soon as new filter is available, and do it with minimal latency
			head = pipeparts.mkgeneric(pipeline, head, "lal_tdwhiten", kernel = sinc_filter[::-1], latency = filter_latency_samples, taper_length = taper_length)
			filter_update = mkadaptivefirfilt(pipeline, filter_update, variable_filter_length = num_samples, adaptive_filter_length = num_samples, update_samples = update_samples, average_samples = average_samples, filter_sample_rate = sample_rate, phase_measurement_frequency = phase_measurement_frequency)
			filter_update.connect("notify::adaptive-filter", update_filter, head, "adaptive_filter", "kernel")
		else:
			# Update filters at specified timestamps to ensure reproducibility
			head = pipeparts.mkgeneric(pipeline, mkqueue(pipeline, head), "lal_tdwhiten", kernel = sinc_filter[::-1], latency = filter_latency_samples, taper_length = taper_length, kernel_endtime = kernel_endtime)
			filter_update = mkadaptivefirfilt(pipeline, filter_update, variable_filter_length = num_samples, adaptive_filter_length = num_samples, update_samples = update_samples, average_samples = average_samples, filter_sample_rate = sample_rate, phase_measurement_frequency = phase_measurement_frequency, filter_timeshift = filter_timeshift)
			filter_update.connect("notify::adaptive-filter", update_filter, head, "adaptive_filter", "kernel")
			filter_update.connect("notify::filter-endtime", update_property_simple, head, "filter_endtime", "kernel_endtime", 1)
	return head

def whiten(pipeline, head, num_samples = 512, nyq_magnitude = 1e15, scale = 'log', td = True):
	# Number of filter samples should be even, since numpy's inverse real fft returns an even length array
	num_samples += num_samples % 2
	fd_num_samples = num_samples // 2 + 1
	fd_filter = numpy.ones(fd_num_samples)
	fd_filter[-1] = nyq_magnitude
	if scale == 'log':
		log_nyq_mag = numpy.log10(nyq_magnitude)
		for i in range(1, fd_num_samples - 1):
			fd_filter[i] = pow(10, log_nyq_mag * float(i) / (fd_num_samples - 1))
	elif scale == 'linear':
		for i in range(1, fd_num_samples - 1):
			fd_filter[i] = nyq_magnitude * float(i) / (fd_num_samples - 1)
	else:
		raise ValueError("calibration_parts.whiten(): scale must be either 'log' or 'linear'.")
		return head

	# Take an inverse fft to get a time-domain filter
	whiten_filter = numpy.fft.irfft(fd_filter)
	# Add delay of half the filter length
	whiten_filter = numpy.roll(whiten_filter, num_samples // 2)
	# Window the filter
	whiten_filter *= numpy.blackman(num_samples)

	# Apply the filter
	return mkcomplexfirbank(pipeline, head, latency = num_samples // 2, fir_matrix = [whiten_filter[::-1]], time_domain = td)

def compute_rms(pipeline, head, rate, average_time, f_min = None, f_max = None, filter_latency = 0.5, rate_out = 16, td = True):
	# Find the root mean square amplitude of a signal between two frequencies
	# Downsample to save computational cost
	head = mkresample(pipeline, head, 4, filter_latency == 0.0, rate)

	# Remove any frequency content we don't care about
	if (f_min is not None) and (f_max is not None):
		head = bandpass(pipeline, head, rate, f_low = f_min, f_high = f_max, filter_latency = filter_latency, td = td)
	elif f_min is not None:
		head = highpass(pipeline, head, rate, fcut = f_min, filter_latency = filter_latency, td = td)
	elif f_max is not None:
		head = lowpass(pipeline, head, rate, fcut = f_max, filter_latency = filter_latency, td = td)

	# Square it
	head = mkpow(pipeline, head, exponent = 2.0)

	# Downsample again to save computational cost
	head = mkresample(pipeline, head, 4, filter_latency == 0.0, rate_out)

	# Compute running average
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", default_kappa_re = 0.0, array_size = 1, avg_array_size = average_time * rate_out, filter_latency = filter_latency)

	# Take the square root
	head = mkpow(pipeline, head, exponent = 0.5)

	return head

#
# Calibration factor related functions
#

def smooth_kappas_no_coherence(pipeline, head, var, expected, N, Nav, default_to_median, filter_latency):
	# Find median of calibration factors array with size N and smooth out medians with an average over Nav samples
	# Use the maximum_offset_re property to determine whether input kappas are good or not
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", maximum_offset_re = var, default_kappa_re = expected, array_size = N, avg_array_size = Nav, default_to_median = default_to_median, filter_latency = filter_latency)
	return head

def smooth_complex_kappas_no_coherence(pipeline, head, real_var, imag_var, real_expected, imag_expected, N, Nav, default_to_median, filter_latency):
	# Find median of complex calibration factors array with size N, split into real and imaginary parts, and smooth out medians with an average over Nav samples
	# Use the maximum_offset_re and maximum_offset_im properties to determine whether input kappas are good or not
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", maximum_offset_re = real_var, maximum_offset_im = imag_var, default_kappa_re = real_expected, default_kappa_im = imag_expected, array_size = N, avg_array_size = Nav, default_to_median = default_to_median, filter_latency = filter_latency)
	return head

def smooth_kappas(pipeline, head, expected, N, Nav, default_to_median, filter_latency):
	# Find median of calibration factors array with size N and smooth out medians with an average over Nav samples
	# Assume input was previously gated with coherence uncertainty to determine if input kappas are good or not
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", default_kappa_re = expected, array_size = N, avg_array_size = Nav, default_to_median = default_to_median, filter_latency = filter_latency)
	return head

def smooth_complex_kappas(pipeline, head, real_expected, imag_expected, N, Nav, default_to_median, filter_latency):
	# Find median of complex calibration factors array with size N and smooth out medians with an average over Nav samples
	# Assume input was previously gated with coherence uncertainty to determine if input kappas are good or not

	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", default_kappa_re = real_expected, default_kappa_im = imag_expected, array_size = N, avg_array_size = Nav, default_to_median = default_to_median, filter_latency = filter_latency)
	return head

def track_bad_kappas_no_coherence(pipeline, head, var, expected, N, Nav, default_to_median, filter_latency):
	# Produce output of 1's or 0's that correspond to median not corrupted (1) or corrupted (0) based on whether median of input array is defualt value.
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", maximum_offset_re = var, default_kappa_re = expected, array_size = N, avg_array_size = Nav if default_to_median else 1, track_bad_kappa = True, default_to_median = default_to_median, filter_latency = filter_latency)
	return head

def track_bad_complex_kappas_no_coherence(pipeline, head, real_var, imag_var, real_expected, imag_expected, N, Nav, default_to_median, filter_latency):
	# Produce output of 1's or 0's that correspond to median not corrupted (1) or corrupted (0) based on whether median of input array is defualt value.
	# Real and imaginary parts are done separately (outputs of lal_smoothkappas can be 1+i, 1, i, or 0)
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", maximum_offset_re = real_var, maximum_offset_im = imag_var, default_kappa_re = real_expected, default_kappa_im = imag_expected, array_size = N, avg_array_size = Nav if default_to_median else 1, track_bad_kappa = True, default_to_median = default_to_median, filter_latency = filter_latency)
	re, im = split_into_real(pipeline, head)
	return re, im

def track_bad_kappas(pipeline, head, expected, N, Nav, default_to_median, filter_latency):
	# Produce output of 1's or 0's that correspond to median not corrupted (1) or corrupted (0) based on whether median of input array is defualt value.
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", default_kappa_re = expected, array_size = N, avg_array_size = Nav if default_to_median else 1, track_bad_kappa = True, default_to_median = default_to_median, filter_latency = filter_latency)
	return head

def track_bad_complex_kappas(pipeline, head, real_expected, imag_expected, N, Nav, default_to_median, filter_latency):
	# Produce output of 1's or 0's that correspond to median not corrupted (1) or corrupted (0) based on whether median of input array is defualt value.
	# Real and imaginary parts are done separately (outputs of lal_smoothkappas can be 1+i, 1, i, or 0)

	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", default_kappa_re = real_expected, default_kappa_im = imag_expected, array_size = N, avg_array_size = Nav if default_to_median else 1, track_bad_kappa = True, default_to_median = default_to_median, filter_latency = filter_latency)
	re, im = split_into_real(pipeline, head)
	return re, im

def smooth_kappas_no_coherence_test(pipeline, head, var, expected, N, Nav, default_to_median, filter_latency):
	# Find median of calibration factors array with size N and smooth out medians with an average over Nav samples
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "raw_kappatst.txt")
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", maximum_offset_re = var, default_kappa_re = expected, array_size = N, avg_array_size = Nav, default_to_median = default_to_median, filter_latency = filter_latency)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "smooth_kappatst.txt")
	return head

def compute_kappa_bits(pipeline, smooth, expected_real, expected_imag, real_ok_var, imag_ok_var, median_samples, avg_samples, status_out_smooth = 1, starting_rate=16, ending_rate=16):

	# Compensate for digital error in the running average
	expected_real_sum = 0.0
	expected_imag_sum = 0.0
	for i in range(0, avg_samples):
		expected_real_sum = expected_real_sum + expected_real
		expected_imag_sum = expected_imag_sum + expected_imag
	expected_real = expected_real_sum / avg_samples
	expected_imag = expected_imag_sum / avg_samples

	# Compute the property bad-data-intervals
	if type(real_ok_var) is not list:
		real_ok_var = [expected_real - real_ok_var, expected_real + real_ok_var]
	if type(imag_ok_var) is not list:
		imag_ok_var = [expected_imag - imag_ok_var, expected_imag + imag_ok_var]
	bad_data_intervals = [real_ok_var[0], imag_ok_var[0], expected_real, expected_imag, expected_real, expected_imag, real_ok_var[1], imag_ok_var[1]]

	# Use lal_insertgap to check if the data is within the required range
	smoothInRange = mkinsertgap(pipeline, smooth, bad_data_intervals = bad_data_intervals, insert_gap = True, remove_gap = False, replace_value = 0, fill_discont = True, block_duration = Gst.SECOND)
	# Turn it into a bit vector
	smoothInRange = pipeparts.mkbitvectorgen(pipeline, smoothInRange, nongap_is_control = True, bit_vector = status_out_smooth)
	smoothInRange = pipeparts.mkcapsfilter(pipeline, smoothInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % starting_rate)
	if starting_rate != ending_rate:
		smoothInRange = pipeparts.mkgeneric(pipeline, smoothInRange, "lal_logicalundersample", required_on = status_out_smooth, status_out = status_out_smooth)
		smoothInRange = pipeparts.mkcapsfilter(pipeline, smoothInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % ending_rate)
	smoothInRangetee = pipeparts.mktee(pipeline, smoothInRange)

	# Require that kappas have been in range for enough time for the smoothing process to settle
	min_samples = int(median_samples / 2) + avg_samples
	smoothInRange = mkgate(pipeline, smoothInRangetee, smoothInRangetee, status_out_smooth, attack_length = -min_samples)
	smoothInRange = pipeparts.mkbitvectorgen(pipeline, smoothInRange, nongap_is_control = True, bit_vector = status_out_smooth)
	smoothInRange = pipeparts.mkcapsfilter(pipeline, smoothInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % ending_rate)

	return smoothInRange

def compute_kappa_bits_only_real(pipeline, smooth, expected, ok_var, median_samples, avg_samples, status_out_smooth = 1, starting_rate=16, ending_rate=16):

	# Compensate for digital error in the running average
	expected_sum = 0.0
	for i in range(0, avg_samples):
		expected_sum = expected_sum + expected
	expected = expected_sum / avg_samples

	if type(ok_var) is list:
		smoothInRange = mkinsertgap(pipeline, smooth, bad_data_intervals = [ok_var[0], expected, expected, ok_var[1]], insert_gap = True, remove_gap = False, replace_value = 0, fill_discont = True, block_duration = Gst.SECOND)
	else:
		smoothInRange = mkinsertgap(pipeline, smooth, bad_data_intervals = [expected - ok_var, expected, expected, expected + ok_var], insert_gap = True, remove_gap = False, replace_value = 0, fill_discont = True, block_duration = Gst.SECOND)
	smoothInRange = pipeparts.mkbitvectorgen(pipeline, smoothInRange, nongap_is_control = True, bit_vector = status_out_smooth)
	smoothInRange = pipeparts.mkcapsfilter(pipeline, smoothInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % starting_rate)
	if starting_rate != ending_rate:
		smoothInRange = pipeparts.mkgeneric(pipeline, smoothInRange, "lal_logicalundersample", required_on = status_out_smooth, status_out = status_out_smooth)
		smoothInRange = pipeparts.mkcapsfilter(pipeline, smoothInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % ending_rate)
	smoothInRangetee = pipeparts.mktee(pipeline, smoothInRange)
	min_samples = int(median_samples / 2) + avg_samples
	smoothInRange = mkgate(pipeline, smoothInRangetee, smoothInRangetee, status_out_smooth, attack_length = -min_samples)
	smoothInRange = pipeparts.mkbitvectorgen(pipeline, smoothInRange, nongap_is_control = True, bit_vector = status_out_smooth)

	return smoothInRange

def merge_into_complex(pipeline, real, imag):
	# Merge real and imag into one complex channel with complex caps
	head = mkinterleave(pipeline, list_srcs(pipeline, real, imag))
	head = pipeparts.mktogglecomplex(pipeline, head)
	return head

def split_into_real(pipeline, complex_chan):
	# split complex channel with complex caps into two channels (real and imag) with real caps

	complex_chan = pipeparts.mktee(pipeline, complex_chan)
	real = pipeparts.mkgeneric(pipeline, complex_chan, "creal")
	imag = pipeparts.mkgeneric(pipeline, complex_chan, "cimag")

#	elem = pipeparts.mkgeneric(pipeline, elem, "deinterleave", keep_positions=True)
#	real = pipeparts.mkgeneric(pipeline, None, "identity")
#	pipeparts.src_deferred_link(elem, "src_0", real.get_static_pad("sink"))
#	imag = pipeparts.mkgeneric(pipeline, None, "identity")
#	pipeparts.src_deferred_link(elem, "src_1", imag.get_static_pad("sink"))

	return real, imag

def complex_audioamplify(pipeline, chan, WR, WI):
	# Multiply a complex channel chan by a complex number WR+I WI
	# Re[out] = -chanI*WI + chanR*WR
	# Im[out] = chanR*WI + chanI*WR

	head = pipeparts.mktogglecomplex(pipeline, chan)
	head = pipeparts.mkmatrixmixer(pipeline, head, matrix=[[WR, WI],[-WI, WR]])
	head = pipeparts.mktogglecomplex(pipeline, head)

	return head

def complex_inverse(pipeline, head):
	# Invert a complex number (1/z)

	head = mkpow(pipeline, head, exponent = -1)

	return head

def complex_division(pipeline, a, b):
	# Perform complex division of c = a/b and output the complex quotient c

	bInv = complex_inverse(pipeline, b)
	c = mkmultiplier(pipeline, list_srcs(pipeline, a, bInv))

	return c

def compute_kappatst_from_filters_file(pipeline, derrfesd, tstexcfesd, pcalfdarm, derrfdarm, ktstfacR, ktstfacI):

	#	       
	# \kappa_TST = ktstfac * (derrfesd/tstexcfesd) * (pcalfdarm/derrfdarm)
	# ktstfac = EP1 = (1/A0fesd) * (C0fdarm/(1+G0fdarm)) * ((1+G0fesd)/C0fesd)
	#

	derrfdarminv = complex_inverse(pipeline, derrfdarm)
	tstexcfesdinv = complex_inverse(pipeline, tstexcfesd)
	ktst = mkmultiplier(pipeline, list_srcs(pipeline, pcalfdarm, derrfdarminv, tstexcfesdinv, derrfesd))
	ktst = complex_audioamplify(pipeline, ktst, ktstfacR, ktstfacI)

	return ktst

def compute_kappatst(pipeline, derrfesd, tstexcfesd, pcalfdarm, derrfdarm, ktstfac):

	#	       
	# \kappa_TST = ktstfac * (derrfesd/tstexcfesd) * (pcalfdarm/derrfdarm)
	# ktstfac = EP1 = (1/A0fesd) * (C0fdarm/(1+G0fdarm)) * ((1+G0fesd)/C0fesd)
	#

	derrfdarminv = complex_inverse(pipeline, derrfdarm)
	tstexcfesdinv = complex_inverse(pipeline, tstexcfesd)
	ktst = mkmultiplier(pipeline, list_srcs(pipeline, ktstfac, pcalfdarm, derrfdarminv, tstexcfesdinv, derrfesd))

	return ktst

def compute_kappapum_from_filters_file(pipeline, derrfpum, pumexcfpum, pcalfpcal, derrfpcal, kpumfacR, kpumfacI):

	#
	# \kappa_PUM = kpumfac * [derr(fpum) / pumexc(fpum)] * [pcal(fpcal) / derr(fpcal)]
	# kpumfac = EP15 = [1 / A_PUM0(fpum)] * [C0(fpcal) / (1 + G0(fpcal))] * [(1 + G0(fpum)) / C0(fpum)]
	#

	pumexcfpuminv = complex_inverse(pipeline, pumexcfpum)
	derrfpcalinv = complex_inverse(pipeline, derrfpcal)
	kpum = mkmultiplier(pipeline, list_srcs(pipeline, derrfpum, pumexcfpuminv, pcalfpcal, derrfpcalinv))
	kpum = complex_audioamplify(pipeline, kpum, kpumfacR, kpumfacI)

	return kpum

def compute_kappapum(pipeline, derrfpum, pumexcfpum, pcalfpcal, derrfpcal, kpumfac):

	#
	# \kappa_PUM = kpumfac * [derr(fpum) / pumexc(fpum)] * [pcal(fpcal) / derr(fpcal)]
	# kpumfac = EP15 = [1 / A_PUM0(fpum)] * [C0(fpcal) / (1 + G0(fpcal))] * [(1 + G0(fpum)) / C0(fpum)]
	#

	pumexcfpuminv = complex_inverse(pipeline, pumexcfpum)
	derrfpcalinv = complex_inverse(pipeline, derrfpcal)
	kpum = mkmultiplier(pipeline, list_srcs(pipeline, kpumfac, derrfpum, pumexcfpuminv, pcalfpcal, derrfpcalinv))

	return kpum

def compute_afctrl_from_filters_file(pipeline, derrfdarm, excfdarm, pcalfpcal, derrfpcal, afctrlfacR, afctrlfacI):

	#
	# A(f_ctrl) = -afctrlfac * (derrfdarm/excfdarm) * (pcalfpcal/derrfpcal)
	# afctrlfac = EP2 = C0fpcal/(1+G0fpcal) * (1+G0fctrl)/C0fctrl
	#

	derrfpcalinv = complex_inverse(pipeline, derrfpcal)
	excfdarminv = complex_inverse(pipeline, excfdarm)
	afctrl = mkmultiplier(pipeline, list_srcs(pipeline, pcalfpcal, derrfpcalinv, excfdarminv, derrfdarm))
	afctrl = complex_audioamplify(pipeline, afctrl, -1.0 * afctrlfacR, -1.0 * afctrlfacI)

	return afctrl
	

def compute_afctrl(pipeline, derrfdarm, excfdarm, pcalfpcal, derrfpcal, afctrlfac):

	#
	# A(f_ctrl) = -afctrlfac * (derrfdarm/excfdarm) * (pcalfpcal/derrfpcal)
	# afctrlfac = EP2 = C0fpcal/(1+G0fpcal) * (1+G0fctrl)/C0fctrl
	#

	derrfpcalinv = complex_inverse(pipeline, derrfpcal)
	excfdarminv = complex_inverse(pipeline, excfdarm)
	afctrl = mkmultiplier(pipeline, list_srcs(pipeline, afctrlfac, pcalfpcal, derrfpcalinv, excfdarminv, derrfdarm))
	afctrl = complex_audioamplify(pipeline, afctrl, -1.0, 0.0)

	return afctrl

def compute_kappauim_from_filters_file(pipeline, EP16R, EP16I, afctrl, ktst, EP4R, EP4I, kpum, EP17R, EP17I):

	#
	# \kappa_uim = EP16 * (afctrl - ktst * EP4 - kpum * EP17)
	#

	kuim = complex_audioamplify(pipeline, mkadder(pipeline, list_srcs(pipeline, afctrl, complex_audioamplify(pipeline, ktst, -1.0 * EP4R, -1.0 * EP4I), complex_audioamplify(pipeline, kpum, -1.0 * EP17R, -1.0 * EP17I))), EP16R, EP16I)

	return kuim

def compute_kappauim(pipeline, EP16, afctrl, ktst, EP4, kpum, EP17):

	#
	# \kappa_uim = EP16 * (afctrl - ktst * EP4 - kpum * EP17)
	#

	ep4_kappatst = mkmultiplier(pipeline, list_srcs(pipeline, ktst, complex_audioamplify(pipeline, EP4, -1.0, 0.0)))
	ep17_kappapum = mkmultiplier(pipeline, list_srcs(pipeline, kpum, complex_audioamplify(pipeline, EP17, -1.0, 0.0)))
	kuim = mkadder(pipeline, list_srcs(pipeline, afctrl, ep4_kappatst, ep17_kappapum))
	kuim = mkmultiplier(pipeline, list_srcs(pipeline, EP16, kuim))

	return kuim

def compute_kappauim_from_filters_file_uim_line(pipeline, derrfuim, uimexcfuim, pcalfpcal, derrfpcal, kuimfacR, kuimfacI):

	#
	# \kappa_UIM = kuimfac * [derr(fuim) / uimexc(fuim)] * [pcal(fpcal) / derr(fpcal)]
	# kuimfac = EP22 = [1 / A_UIM0(fuim)] * [C0(fpcal) / (1 + G0(fpcal))] * [(1 + G0(fuim)) / C0(fuim)]
	#

	uimexcfuiminv = complex_inverse(pipeline, uimexcfuim)
	derrfpcalinv = complex_inverse(pipeline, derrfpcal)
	kuim = mkmultiplier(pipeline, list_srcs(pipeline, derrfuim, uimexcfuiminv, pcalfpcal, derrfpcalinv))
	kuim = complex_audioamplify(pipeline, kuim, kuimfacR, kuimfacI)

	return kuim

def compute_kappauim_uim_line(pipeline, derrfuim, uimexcfuim, pcalfpcal, derrfpcal, kuimfac):

	#
	# \kappa_UIM = kuimfac * [derr(fuim) / uimexc(fuim)] * [pcal(fpcal) / derr(fpcal)]
	# kuimfac = EP22 = [1 / A_UIM0(fuim)] * [C0(fpcal) / (1 + G0(fpcal))] * [(1 + G0(fuim)) / C0(fuim)]
	#

	uimexcfuiminv = complex_inverse(pipeline, uimexcfuim)
	derrfpcalinv = complex_inverse(pipeline, derrfpcal)
	kuim = mkmultiplier(pipeline, list_srcs(pipeline, kuimfac, derrfuim, uimexcfuiminv, pcalfpcal, derrfpcalinv))

	return kuim

def compute_kappapu_from_filters_file(pipeline, EP3R, EP3I, afctrl, ktst, EP4R, EP4I):

	#
	# \kappa_pu = EP3 * (afctrl - ktst * EP4)
	#

	kpu = complex_audioamplify(pipeline, mkadder(pipeline, list_srcs(pipeline, afctrl, complex_audioamplify(pipeline, ktst, -1.0*EP4R, -1.0*EP4I))), EP3R, EP3I)	

	return kpu

def compute_kappapu(pipeline, EP3, afctrl, ktst, EP4):
	
	#
	# \kappa_pu = EP3 * (afctrl - ktst * EP4)
	#

	ep4_kappatst = mkmultiplier(pipeline, list_srcs(pipeline, ktst, complex_audioamplify(pipeline, EP4, -1.0, 0.0)))
	afctrl_kappatst = mkadder(pipeline, list_srcs(pipeline, afctrl, ep4_kappatst))
	kpu = mkmultiplier(pipeline, list_srcs(pipeline, EP3, afctrl_kappatst))

	return kpu

def compute_kappaa_from_filters_file(pipeline, afctrl, EP4R, EP4I, EP5R, EP5I):

	#
	#\kappa_a = afctrl / (EP4+EP5)
	#

	facR = (EP4R + EP5R) / ((EP4R + EP5R)**2 + (EP4I + EP5I)**2)
	facI = -(EP4I + EP5I) / ((EP4R + EP5R)**2 + (EP4I + EP5I)**2)

	ka = complex_audioamplify(pipeline, afctrl, facR, facI) 

	return ka

def compute_kappaa(pipeline, afctrl, EP4, EP5):

	#
	#\kappa_a = afctrl / (EP4 + EP5)
	#

	ka = complex_division(pipeline, afctrl, mkadder(pipeline, list_srcs(pipeline, EP4, EP5)))

	return ka

def compute_exact_kappas_from_filters_file(pipeline, X, freqs, EPICS, rate, default_fcc = 400, default_fs_squared = 1.0, default_fs_over_Q = 1.0):

	#
	# See P1900052, Section 5.2.6 for details.  All constants are contained in the list
	# variable EPICS.  The variable freqs is a list containing calibration line
	# frequencies, stored in the order f1, f2, fT, fP, fU, i.e., the Pcal lines come
	# first, and then the actuator lines.  All other quantities evaluated at the
	# calibration lines are stored in the same order.  The list variable X contains the
	# ratios X[i] = injection(f_i) / d_err(f_i) for each calibration line frequency.
	#

	kappas = []
	num_lines = len(freqs)
	num_stages = num_lines - 2 # Stages of actuation (currently 3)
	MV_matrix = list(numpy.zeros(2 * num_stages * (2 * num_stages + 1)))
	Y = []
	Yreal = []
	Yimag = []
	CAX = []
	CAXreal = []
	CAXimag = []
	Gres = []
	kappas = []

	for i in range(num_lines):
		if i < 2:
			# Then it's a Pcal line
			Y.append(pipeparts.mktee(pipeline, complex_audioamplify(pipeline, X[i], EPICS[2 * (1 + num_stages) * i], EPICS[2 * (1 + num_stages) * i + 1])))
			Yreal.append(pipeparts.mktee(pipeline, mkcapsfiltersetter(pipeline, pipeparts.mkgeneric(pipeline, Y[i], "creal"), "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate, name = "capsfilter_Yreal_%d" % i)))
			Yimag.append(pipeparts.mktee(pipeline, mkcapsfiltersetter(pipeline, pipeparts.mkgeneric(pipeline, Y[i], "cimag"), "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate, name = "capsfilter_Yimag_%d" % i)))
		else:
			# It's an actuator line
			CAX.append(pipeparts.mktee(pipeline, complex_audioamplify(pipeline, X[i], EPICS[2 * (1 + num_stages) * i], EPICS[2 * (1 + num_stages) * i + 1])))
			CAXreal.append(pipeparts.mktee(pipeline, mkcapsfiltersetter(pipeline, pipeparts.mkgeneric(pipeline, CAX[-1], "creal"), "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate, name = "capsfilter_CAXreal_%d" % i)))
			CAXimag.append(pipeparts.mktee(pipeline, mkcapsfiltersetter(pipeline, pipeparts.mkgeneric(pipeline, CAX[-1], "cimag"), "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate, name = "capsfilter_CAXimag_%d" % i)))

	# Let's start by computing the V's of Eqs. 5.2.78 and 5.2.79
	for j in range(num_stages):
		factor1 = pow(freqs[0], -2) - pow(freqs[2 + j], -2)
		factor2 = pow(freqs[2 + j], -2) - pow(freqs[1], -2)
		factor3 = freqs[1] * (pow(freqs[0], 2) - pow(freqs[2 + j], 2))
		factor4 = freqs[0] * (pow(freqs[2 + j], 2) - pow(freqs[1], 2))
		Vj = mkadder(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, pipeparts.mkcapsfilter(pipeline, Yreal[1], "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate), factor1), pipeparts.mkaudioamplify(pipeline, pipeparts.mkcapsfilter(pipeline, Yreal[0], "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate), factor2)))
		Vj = pipeparts.mkcapsfilter(pipeline, Vj, "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate)
		Vjplus3 = mkadder(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, pipeparts.mkcapsfilter(pipeline, Yimag[1], "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate), factor3), pipeparts.mkaudioamplify(pipeline, pipeparts.mkcapsfilter(pipeline, Yimag[0], "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate), factor4)))
		Vjplus3 = pipeparts.mkcapsfilter(pipeline, Vjplus3, "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate)
		MV_matrix[j] = Vj
		MV_matrix[num_stages + j] = Vjplus3

	# Now let's compute the elements of the matrix M, given by Eqs. 5.2.70 - 5.2.77
	# Many of the elements are constant, so make a stream of ones to multiply
	if num_stages > 1:
		ones = pipeparts.mktee(pipeline, mkpow(pipeline, pipeparts.mkcapsfilter(pipeline, Yreal[0], "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate), exponent = 0.0))
	for j in range(num_stages):
		# Time-dependent matrix elements
		factor = pow(freqs[0], -2) - pow(freqs[1], -2)
		addend = (pow(freqs[0], -2) - pow(freqs[2 + j], -2)) * EPICS[2 * ((1 + num_stages) + 1 + j)] + (pow(freqs[2 + j], -2) - pow(freqs[1], -2)) * EPICS[2 * (1 + j)] - (pow(freqs[0], -2) - pow(freqs[1], -2)) * EPICS[2 * ((2 + j) * (1 + num_stages) + 1 + j)]
		Mjj = pipeparts.mkgeneric(pipeline, pipeparts.mkaudioamplify(pipeline, pipeparts.mkcapsfilter(pipeline, CAXreal[j], "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate), factor), "lal_add_constant", value = addend)
		factor = -2.0 * numpy.pi * freqs[2 + j] * (pow(freqs[0], -2) - pow(freqs[1], -2))
		addend = -2.0 * numpy.pi * freqs[1] * (pow(freqs[0], -2) - pow(freqs[2 + j], -2)) * EPICS[1 + 2 * ((1 + num_stages) + 1 + j)] - 2.0 * numpy.pi * freqs[0] * (pow(freqs[2 + j], -2) - pow(freqs[1], -2)) * EPICS[1 + 2 * (1 + j)] + 2.0 * numpy.pi * freqs[2 + j] * (pow(freqs[0], -2) - pow(freqs[1], -2)) * EPICS[1 + 2 * ((2 + j) * (1 + num_stages) + 1 + j)]
		Mjjplus3 = pipeparts.mkgeneric(pipeline, pipeparts.mkaudioamplify(pipeline, pipeparts.mkcapsfilter(pipeline, CAXimag[j], "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate), factor), "lal_add_constant", value = addend)
		factor = -freqs[2 + j] * (pow(freqs[1], 2) - pow(freqs[0], 2))
		addend = freqs[1] * (pow(freqs[0], 2) - pow(freqs[2 + j], 2)) * EPICS[1 + 2 * ((1 + num_stages) + 1 + j)] + freqs[0] * (pow(freqs[2 + j], 2) - pow(freqs[1], 2)) * EPICS[1 + 2 * (1 + j)] + freqs[2 + j] * (pow(freqs[1], 2) - pow(freqs[0], 2)) * EPICS[1 + 2 * ((2 + j) * (1 + num_stages) + 1 + j)]
		Mjplus3j = pipeparts.mkgeneric(pipeline, pipeparts.mkaudioamplify(pipeline, pipeparts.mkcapsfilter(pipeline, CAXimag[j], "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate), factor), "lal_add_constant", value = addend)
		factor = -2.0 * numpy.pi * pow(freqs[2 + j], 2) * (pow(freqs[1], 2) - pow(freqs[0], 2))
		addend = 2.0 * numpy.pi * pow(freqs[1], 2) * (pow(freqs[0], 2) - pow(freqs[2 + j], 2)) * EPICS[2 * ((1 + num_stages) + 1 + j)] + 2.0 * numpy.pi * pow(freqs[0], 2) * (pow(freqs[2 + j], 2) - pow(freqs[1], 2)) * EPICS[2 * (1 + j)] + 2.0 * numpy.pi * pow(freqs[2 + j], 2) * (pow(freqs[1], 2) - pow(freqs[0], 2)) * EPICS[2 * ((2 + j) * (1 + num_stages) + 1 + j)]
		Mjplus3jplus3 = pipeparts.mkgeneric(pipeline, pipeparts.mkaudioamplify(pipeline, pipeparts.mkcapsfilter(pipeline, CAXreal[j], "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate), factor), "lal_add_constant", value = addend)
		# Add these into the matrix
		MV_matrix[(1 + j) * 2 * num_stages + j] = Mjj
		MV_matrix[(1 + j) * 2 * num_stages + num_stages + j] = Mjjplus3
		MV_matrix[(1 + num_stages + j) * 2 * num_stages + j] = Mjplus3j
		MV_matrix[(1 + num_stages + j) * 2 * num_stages + num_stages + j] = Mjplus3jplus3

		# Constant matrix elements
		knotequalj = list(numpy.arange(num_stages))
		knotequalj.remove(j)
		for k in knotequalj:
			factor = (pow(freqs[0], -2) - pow(freqs[2 + j], -2)) * EPICS[2 * ((1 + num_stages) + 1 + k)] + (pow(freqs[2 + j], -2) - pow(freqs[1], -2)) * EPICS[2 * (1 + k)] - (pow(freqs[0], -2) - pow(freqs[1], -2)) * EPICS[2 * ((2 + j) * (1 + num_stages) + 1 + k)]
			Mjk = pipeparts.mkaudioamplify(pipeline, pipeparts.mkcapsfilter(pipeline, ones, "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate), factor)
			factor = -2.0 * numpy.pi * freqs[1] * (pow(freqs[0], -2) - pow(freqs[2 + j], -2)) * EPICS[1 + 2 * ((1 + num_stages) + 1 + k)] - 2.0 * numpy.pi * freqs[0] * (pow(freqs[2 + j], -2) - pow(freqs[1], -2)) * EPICS[1 + 2 * (1 + k)] + 2.0 * numpy.pi * freqs[2 + j] * (pow(freqs[0], -2) - pow(freqs[1], -2)) * EPICS[1 + 2 * ((2 + j) * (1 + num_stages) + 1 + k)]
			Mjkplus3 = pipeparts.mkaudioamplify(pipeline, pipeparts.mkcapsfilter(pipeline, ones, "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate), factor)
			factor = freqs[1] * (pow(freqs[0], 2) - pow(freqs[2 + j], 2)) * EPICS[1 + 2 * ((1 + num_stages) + 1 + k)] + freqs[0] * (pow(freqs[2 + j], 2) - pow(freqs[1], 2)) * EPICS[1 + 2 * (1 + k)] + freqs[2 + j] * (pow(freqs[1], 2) - pow(freqs[0], 2)) * EPICS[1 + 2 * ((2 + j) * (1 + num_stages) + 1 + k)]
			Mjplus3k = pipeparts.mkaudioamplify(pipeline, pipeparts.mkcapsfilter(pipeline, ones, "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate), factor)
			factor = 2.0 * numpy.pi * pow(freqs[1], 2) * (pow(freqs[0], 2) - pow(freqs[2 + j], 2)) * EPICS[2 * ((1 + num_stages) + 1 + k)] + 2.0 * numpy.pi * pow(freqs[0], 2) * (pow(freqs[2 + j], 2) - pow(freqs[1], 2)) * EPICS[2 * (1 + k)] + 2.0 * numpy.pi * pow(freqs[2 + j], 2) * (pow(freqs[1], 2) - pow(freqs[0], 2)) * EPICS[2 * ((2 + j) * (1 + num_stages) + 1 + k)]
			Mjplus3kplus3 = pipeparts.mkaudioamplify(pipeline, pipeparts.mkcapsfilter(pipeline, ones, "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate), factor)
			# Add these into the matrix
			MV_matrix[(1 + j) * 2 * num_stages + k] = Mjk
			MV_matrix[(1 + j) * 2 * num_stages + num_stages + k] = Mjkplus3
			MV_matrix[(1 + num_stages + j) * 2 * num_stages + k] = Mjplus3k
			MV_matrix[(1 + num_stages + j) * 2 * num_stages + num_stages + k] = Mjplus3kplus3

	# Now pass these to the matrix solver to find kappa_T, kappa_P, kappa_U, tau_T, tau_P, and tau_U.
	MV_matrix = mkinterleave(pipeline, MV_matrix)
	MV_matrix = pipeparts.mkcapsfilter(pipeline, MV_matrix, "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=%d" % (rate, (2 * num_stages) * (2 * num_stages + 1)))
	kappas = pipeparts.mkgeneric(pipeline, MV_matrix, "lal_matrixsolver")
	kappas = list(mkdeinterleave(pipeline, pipeparts.mkcapsfilter(pipeline, kappas, "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=%d" % (rate, 2 * num_stages)), 2 * num_stages))
	for i in range(len(kappas)):
		kappas[i] = pipeparts.mkcapsfilter(pipeline, kappas[i], "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate)
		if i >= len(kappas) // 2:
			kappas[i] = mkmultiplier(pipeline, [kappas[i], mkpow(pipeline, kappas[i - len(kappas) // 2], exponent = -1.0)])
		kappas[i] = pipeparts.mktee(pipeline, kappas[i])

	# Next, compute kappa_C.  This is going to take some work...
	# Start by computing G_res at each frequency, defined in Eq. 5.2.30
	for n in range(2):
		Gres_components = []
		for j in range(num_stages):
			kappajGresjatn = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, pipeparts.mkcapsfilter(pipeline, kappas[j], "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate), matrix = [[EPICS[2 * (n * (1 + num_stages) + 1 + j)], EPICS[1 + 2 * (n * (1 + num_stages) + 1 + j)]]]))
			i_omega_tau = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, pipeparts.mkcapsfilter(pipeline, kappas[num_stages + j], "audio/x-raw,format=F64LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate), matrix = [[0, 2.0 * numpy.pi * freqs[n]]]))
			i_omega_tau = mkcapsfiltersetter(pipeline, i_omega_tau, "audio/x-raw,format=Z128LE,rate=%d,channel-mask=(bitmask)0x0,channels=1" % rate)
			phase = pipeparts.mkgeneric(pipeline, i_omega_tau, "cexp")
			Gres_components.append(mkmultiplier(pipeline, list_srcs(pipeline, kappajGresjatn, phase)))
		Gres.append(mkadder(pipeline, Gres_components))

	sensing_inputs = mkinterleave(pipeline, Gres + Y, complex_data = True)
	sensing_outputs = pipeparts.mkgeneric(pipeline, sensing_inputs, "lal_sensingtdcfs", sensing_model = 0, freq1 = freqs[0], freq2 = freqs[1], current_fcc = default_fcc, current_fs_squared = default_fs_squared, current_fs_over_Q = default_fs_over_Q)
	sensing_outputs = list(mkdeinterleave(pipeline, sensing_outputs, 4))

	kappas += sensing_outputs

	return kappas

def compute_S_from_filters_file(pipeline, EP6R, EP6I, pcalfpcal2, derrfpcal2, EP7R, EP7I, ktst, EP8R, EP8I, kpu, EP9R, EP9I):

	#	
	# S = 1/EP6 * ( pcalfpcal2/derrfpcal2 - EP7*(ktst*EP8 + kpu*EP9) ) ^ (-1)
	#

	pcal_over_derr = complex_division(pipeline, pcalfpcal2, derrfpcal2)
	ep8_kappatst = complex_audioamplify(pipeline, ktst, EP8R, EP8I)
	ep9_kappapu = complex_audioamplify(pipeline, kpu, EP9R, EP9I)
	kappatst_kappapu = mkadder(pipeline, list_srcs(pipeline, ep8_kappatst, ep9_kappapu))
	kappatst_kappapu = complex_audioamplify(pipeline, kappatst_kappapu,  -1.0*EP7R, -1.0*EP7I)
	Sinv = mkadder(pipeline, list_srcs(pipeline, pcal_over_derr, kappatst_kappapu))
	Sinv = complex_audioamplify(pipeline, Sinv, EP6R, EP6I)
	S = complex_inverse(pipeline, Sinv)
	
	return S

def compute_S_from_filters_file_split_act(pipeline, fpcal2, EP6R, EP6I, pcalfpcal2, derrfpcal2, EP7R, EP7I, ftst, ktst, apply_complex_ktst, EP8R, EP8I, fpum, kpum, apply_complex_kpum, EP18R, EP18I, fuim, kuim, apply_complex_kuim, EP19R, EP19I):

	#       
	# S = (1 / EP6) * (pcalfpcal2 / derrfpcal2 - EP7 * (ktst * EP8 + kpum * EP18 + kuim * EP19))^(-1)
	#

	if apply_complex_ktst:
		ktst = pipeparts.mkgeneric(pipeline, ktst, "lpshiftfreq", frequency_ratio = fpcal2 / ftst)
		ep8_ktst = complex_audioamplify(pipeline, ktst, EP8R, EP8I)
	else:
		ep8_ktst = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, pipeparts.mkgeneric(pipeline, ktst, "cabs"), matrix = [[EP8R, EP8I]]))
	if apply_complex_kpum:
		kpum = pipeparts.mkgeneric(pipeline, kpum, "lpshiftfreq", frequency_ratio = fpcal2 / fpum)
		ep18_kpum = complex_audioamplify(pipeline, kpum, EP18R, EP18I)
	else:
		ep18_kpum = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, pipeparts.mkgeneric(pipeline, kpum, "cabs"), matrix = [[EP18R, EP18I]]))
	if apply_complex_kuim:
		kuim = pipeparts.mkgeneric(pipeline, kuim, "lpshiftfreq", frequency_ratio = fpcal2 / fuim)
		ep19_kuim = complex_audioamplify(pipeline, kuim, EP19R, EP19I)
	else:
		ep19_kuim = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, pipeparts.mkgeneric(pipeline, kuim, "cabs"), matrix = [[EP19R, EP19I]]))

	pcal_over_derr = complex_division(pipeline, pcalfpcal2, derrfpcal2)
	A_at_fpcal2 = mkadder(pipeline, list_srcs(pipeline, ep8_ktst, ep18_kpum, ep19_kuim))
	DA_at_fpcal2 = complex_audioamplify(pipeline, A_at_fpcal2,  -1.0 * EP7R, -1.0 * EP7I)
	Sinv = mkadder(pipeline, list_srcs(pipeline, pcal_over_derr, DA_at_fpcal2))
	Sinv = complex_audioamplify(pipeline, Sinv, EP6R, EP6I)
	S = complex_inverse(pipeline, Sinv)

	return S

def compute_S(pipeline, EP6, pcalfpcal2, derrfpcal2, EP7, ktst, EP8, kpu, EP9):

	#	
	# S = 1/EP6 * ( pcalfpcal2/derrfpcal2 - EP7*(ktst*EP8 + kpum*EP9) ) ^ (-1)
	#

	pcal_over_derr = complex_division(pipeline, pcalfpcal2, derrfpcal2)
	ep8_kappatst = mkmultiplier(pipeline, list_srcs(pipeline, ktst, EP8))
	ep9_kappapu = mkmultiplier(pipeline, list_srcs(pipeline, kpu, EP9))
	kappatst_kappapu = mkadder(pipeline, list_srcs(pipeline, ep8_kappatst, ep9_kappapu))
	kappatst_kappapu = mkmultiplier(pipeline, list_srcs(pipeline, complex_audioamplify(pipeline, EP7, -1.0, 0.0), kappatst_kappapu))
	Sinv = mkadder(pipeline, list_srcs(pipeline, pcal_over_derr, kappatst_kappapu))
	Sinv = mkmultiplier(pipeline, list_srcs(pipeline, EP6, Sinv))
	S = complex_inverse(pipeline, Sinv)

	return S
def compute_S_split_act(pipeline, fpcal2, EP6, pcalfpcal2, derrfpcal2, EP7, ftst, ktst, apply_complex_ktst, EP8, fpum, kpum, apply_complex_kpum, EP18, fuim, kuim, apply_complex_kuim, EP19):

	#       
	# S = (1 / EP6) * (pcalfpcal2 / derrfpcal2 - EP7 * (ktst * EP8 + kpu * EP18 + kuim * EP19))^(-1)
	#

	if apply_complex_ktst:
		ktst = pipeparts.mkgeneric(pipeline, ktst, "lpshiftfreq", frequency_ratio = fpcal2 / ftst)
	else:
		ktst = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, pipeparts.mkgeneric(pipeline, ktst, "cabs"), matrix = [[1.0, 0.0]]))
	if apply_complex_kpum:
		kpum = pipeparts.mkgeneric(pipeline, kpum, "lpshiftfreq", frequency_ratio = fpcal2 / fpum)
	else:
		kpum = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, pipeparts.mkgeneric(pipeline, kpum, "cabs"), matrix = [[1.0, 0.0]]))
	if apply_complex_kuim:
		kuim = pipeparts.mkgeneric(pipeline, kuim, "lpshiftfreq", frequency_ratio = fpcal2 / fuim)
	else:
		kuim = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, pipeparts.mkgeneric(pipeline, kuim, "cabs"), matrix = [[1.0, 0.0]]))

	pcal_over_derr = complex_division(pipeline, pcalfpcal2, derrfpcal2)
	ep8_ktst = mkmultiplier(pipeline, list_srcs(pipeline, ktst, EP8))
	ep18_kpum = mkmultiplier(pipeline, list_srcs(pipeline, kpum, EP18))
	ep19_kuim = mkmultiplier(pipeline, list_srcs(pipeline, kuim, EP19))
	A_at_fpcal2 = mkadder(pipeline, list_srcs(pipeline, ep8_ktst, ep18_kpum, ep19_kuim))
	DA_at_fpcal2 = mkmultiplier(pipeline, list_srcs(pipeline, complex_audioamplify(pipeline, EP7, -1.0, 0.0), A_at_fpcal2))
	Sinv = mkadder(pipeline, list_srcs(pipeline, pcal_over_derr, DA_at_fpcal2))
	Sinv = mkmultiplier(pipeline, list_srcs(pipeline, EP6, Sinv))
	S = complex_inverse(pipeline, Sinv)

	return S

def compute_kappac(pipeline, SR, SI):

	#
	# \kappa_C = |S|^2 / Re[S]
	#

	SR = pipeparts.mktee(pipeline, SR)
	S2 = mkadder(pipeline, list_srcs(pipeline, mkpow(pipeline, SR, exponent=2.0), mkpow(pipeline, SI, exponent=2.0)))
	kc = mkmultiplier(pipeline, list_srcs(pipeline, S2, mkpow(pipeline, SR, exponent=-1.0)))
	return kc

def compute_fcc(pipeline, SR, SI, fpcal2, freq_update = None):

	#
	# f_cc = - (Re[S]/Im[S]) * fpcal2
	#

	
	fcc = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, SR, -1.0), mkpow(pipeline, SI, exponent=-1.0)))
	fcc = pipeparts.mkaudioamplify(pipeline, fcc, fpcal2)
	if freq_update is not None:
		freq_update.connect("notify::timestamped-average", update_timestamped_property, fcc, "timestamped_average", "amplification", 1)
	return fcc

def compute_Xi_from_filters_file(pipeline, pcalfpcal4, darmfpcal4, fpcal4, EP11_real, EP11_imag, EP12_real, EP12_imag, EP13_real, EP13_imag, EP14_real, EP14_imag, ktst, kpu, kc, fcc):

	#
	# Xi = -1 + ((EP11*kc) / (1 + i * f_src/f_cc)) * (pcalfpcal4/derrfpcal4 - EP12*(ktst*EP13 + kpu*EP14))
	#

	Atst = complex_audioamplify(pipeline, ktst, EP13_real, EP13_imag)
	Apu = complex_audioamplify(pipeline, kpu, EP14_real, EP14_imag) 
	A = mkadder(pipeline, list_srcs(pipeline, Atst, Apu))
	minusAD = complex_audioamplify(pipeline, A, -1.0 * EP12_real, -1.0 * EP12_imag)
	pcal_over_derr = complex_division(pipeline, pcalfpcal4, darmfpcal4)
	pcal_over_derr_res = mkadder(pipeline, list_srcs(pipeline, pcal_over_derr, minusAD))
	fpcal4_over_fcc = pipeparts.mkaudioamplify(pipeline, mkpow(pipeline, fcc, exponent = -1.0), fpcal4)
	i_fpcal4_over_fcc = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, fpcal4_over_fcc, matrix = [[0, 1]]))
	i_fpcal4_over_fcc_plus_one = pipeparts.mkgeneric(pipeline, i_fpcal4_over_fcc, "lal_add_constant", value = 1.0)
	i_fpcal4_over_fcc_plus_one_inv = complex_inverse(pipeline, i_fpcal4_over_fcc_plus_one)
	kc_EP11 = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, kc, matrix = [[EP11_real, EP11_imag]]))
	Xi_plus_one = mkmultiplier(pipeline, list_srcs(pipeline, kc_EP11, i_fpcal4_over_fcc_plus_one_inv, pcal_over_derr_res))
	Xi = pipeparts.mkgeneric(pipeline, Xi_plus_one, "lal_add_constant", value = -1.0)

	return Xi

def compute_Xi_from_filters_file_split_act(pipeline, pcalfpcal4, darmfpcal4, fpcal4, EP11R, EP11I, EP12R, EP12I, EP13R, EP13I, EP20R, EP20I, EP21R, EP21I, ftst, ktst, apply_complex_ktst, fpum, kpum, apply_complex_kpum, fuim, kuim, apply_complex_kuim, kc, fcc):

	#
	# Xi = -1 + ((EP11 * kc) / (1 + i * f_src / f_cc)) * (pcalfpcal4 / derrfpcal4 - EP12 * (ktst * EP13 + kpum * EP20 + kuim * EP21))
	#

	if apply_complex_ktst:
		ktst = pipeparts.mkgeneric(pipeline, ktst, "lpshiftfreq", frequency_ratio = fpcal4 / ftst)
		Atst = complex_audioamplify(pipeline, ktst, EP13R, EP13I)
	else:
		Atst = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, pipeparts.mkgeneric(pipeline, ktst, "cabs"), matrix = [[EP13R, EP13I]]))
	if apply_complex_kpum:
		kpum = pipeparts.mkgeneric(pipeline, kpum, "lpshiftfreq", frequency_ratio = fpcal4 / fpum)
		Apum = complex_audioamplify(pipeline, kpum, EP20R, EP20I)
	else:
		Apum = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, pipeparts.mkgeneric(pipeline, kpum, "cabs"), matrix = [[EP20R, EP20I]]))
	if apply_complex_kuim:
		kuim = pipeparts.mkgeneric(pipeline, kuim, "lpshiftfreq", frequency_ratio = fpcal4 / fuim)
		Auim = complex_audioamplify(pipeline, kuim, EP21R, EP21I)
	else:
		Auim = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, pipeparts.mkgeneric(pipeline, kuim, "cabs"), matrix = [[EP21R, EP21I]]))

	A = mkadder(pipeline, list_srcs(pipeline, Atst, Apum, Auim))
	minusAD = complex_audioamplify(pipeline, A, -1.0 * EP12R, -1.0 * EP12I)
	pcal_over_derr = complex_division(pipeline, pcalfpcal4, darmfpcal4)
	pcal_over_derr_res = mkadder(pipeline, list_srcs(pipeline, pcal_over_derr, minusAD))
	fpcal4_over_fcc = pipeparts.mkaudioamplify(pipeline, mkpow(pipeline, fcc, exponent = -1.0), fpcal4)
	i_fpcal4_over_fcc = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, fpcal4_over_fcc, matrix = [[0, 1]]))
	i_fpcal4_over_fcc_plus_one = pipeparts.mkgeneric(pipeline, i_fpcal4_over_fcc, "lal_add_constant", value = 1.0)
	i_fpcal4_over_fcc_plus_one_inv = complex_inverse(pipeline, i_fpcal4_over_fcc_plus_one)
	kc_EP11 = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, kc, matrix = [[EP11R, EP11I]]))
	Xi_plus_one = mkmultiplier(pipeline, list_srcs(pipeline, kc_EP11, i_fpcal4_over_fcc_plus_one_inv, pcal_over_derr_res))
	Xi = pipeparts.mkgeneric(pipeline, Xi_plus_one, "lal_add_constant", value = -1.0)

	return Xi

def compute_Xi(pipeline, pcalfpcal4, darmfpcal4, fpcal4, EP11, EP12, EP13, EP14, ktst, kpu, kc, fcc):

	#
	# Xi = -1 + ((EP11*kc) / (1 + i * f_src/f_cc)) * (pcalfpcal4/derrfpcal4 - EP12*(ktst*EP13 + kpu*EP14))
	#

	complex_kc = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, kc, matrix=[[1,0]]))
	Atst = mkmultiplier(pipeline, list_srcs(pipeline, EP13, ktst))
	Apu = mkmultiplier(pipeline, list_srcs(pipeline, EP14, kpu))
	A = mkadder(pipeline, list_srcs(pipeline, Atst, Apu))
	minusAD = mkmultiplier(pipeline, list_srcs(pipeline, complex_audioamplify(pipeline, EP12, -1.0, 0.0), A))
	pcal_over_derr = complex_division(pipeline, pcalfpcal4, darmfpcal4)
	pcal_over_derr_res = mkadder(pipeline, list_srcs(pipeline, pcal_over_derr, minusAD))
	fpcal4_over_fcc = pipeparts.mkaudioamplify(pipeline, mkpow(pipeline, fcc, exponent = -1.0), fpcal4)
	i_fpcal4_over_fcc = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, fpcal4_over_fcc, matrix = [[0, 1]]))
	i_fpcal4_over_fcc_plus_one = pipeparts.mkgeneric(pipeline, i_fpcal4_over_fcc, "lal_add_constant", value = 1.0)
	i_fpcal4_over_fcc_plus_one_inv = complex_inverse(pipeline, i_fpcal4_over_fcc_plus_one)
	Xi_plus_one = mkmultiplier(pipeline, list_srcs(pipeline, EP11, complex_kc, i_fpcal4_over_fcc_plus_one_inv, pcal_over_derr_res))
	Xi = pipeparts.mkgeneric(pipeline, Xi_plus_one, "lal_add_constant", value = -1.0)

	return Xi

def compute_Xi_split_act(pipeline, pcalfpcal4, darmfpcal4, fpcal4, EP11, EP12, EP13, EP20, EP21, ftst, ktst, apply_complex_ktst, fpum, kpum, apply_complex_kpum, fuim, kuim, apply_complex_kuim, kc, fcc):

	#
	# Xi = -1 + ((EP11 * kc) / (1 + i * f_src / f_cc)) * (pcalfpcal4 / derrfpcal4 - EP12 * (ktst * EP13 + kpum * EP20 + kuim * EP21))
	#

	if apply_complex_ktst:
		ktst = pipeparts.mkgeneric(pipeline, ktst, "lpshiftfreq", frequency_ratio = fpcal4 / ftst)
	else:
		ktst = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, pipeparts.mkgeneric(pipeline, ktst, "cabs"), matrix = [[1.0, 0.0]]))
	if apply_complex_kpum:
		kpum = pipeparts.mkgeneric(pipeline, kpum, "lpshiftfreq", frequency_ratio = fpcal4 / fpum)
	else:
		kpum = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, pipeparts.mkgeneric(pipeline, kpum, "cabs"), matrix = [[1.0, 0.0]]))
	if apply_complex_kuim:
		kuim = pipeparts.mkgeneric(pipeline, kuim, "lpshiftfreq", frequency_ratio = fpcal4 / fuim)
	else:
		kuim = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, pipeparts.mkgeneric(pipeline, kuim, "cabs"), matrix = [[1.0, 0.0]]))

	complex_kc = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, kc, matrix=[[1,0]]))
	Atst = mkmultiplier(pipeline, list_srcs(pipeline, EP13, ktst))
	Apum = mkmultiplier(pipeline, list_srcs(pipeline, EP20, kpum))
	Auim = mkmultiplier(pipeline, list_srcs(pipeline, EP21, kuim))
	A = mkadder(pipeline, list_srcs(pipeline, Atst, Apum, Auim))
	minusAD = mkmultiplier(pipeline, list_srcs(pipeline, complex_audioamplify(pipeline, EP12, -1.0, 0.0), A))
	pcal_over_derr = complex_division(pipeline, pcalfpcal4, darmfpcal4)
	pcal_over_derr_res = mkadder(pipeline, list_srcs(pipeline, pcal_over_derr, minusAD))
	fpcal4_over_fcc = pipeparts.mkaudioamplify(pipeline, mkpow(pipeline, fcc, exponent = -1.0), fpcal4)
	i_fpcal4_over_fcc = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, fpcal4_over_fcc, matrix = [[0, 1]]))
	i_fpcal4_over_fcc_plus_one = pipeparts.mkgeneric(pipeline, i_fpcal4_over_fcc, "lal_add_constant", value = 1.0)
	i_fpcal4_over_fcc_plus_one_inv = complex_inverse(pipeline, i_fpcal4_over_fcc_plus_one)
	Xi_plus_one = mkmultiplier(pipeline, list_srcs(pipeline, EP11, complex_kc, i_fpcal4_over_fcc_plus_one_inv, pcal_over_derr_res))
	Xi = pipeparts.mkgeneric(pipeline, Xi_plus_one, "lal_add_constant", value = -1.0)

	return Xi

def compute_uncertainty_reduction(pipeline, head, demod_samples, median_samples, avg_samples):
	#
	# How much is the uncertainty of the TDCFs reduced by the running median
	# and average, given the length of the demodulation filter?
	#

	# Represent each process as a filter with the same effect on uncertainty
	demod_filt = fir.kaiser(demod_samples, 3 * numpy.pi)
	demod_filt /= numpy.sum(demod_filt)
	if demod_samples < 1:
		demod_filt = numpy.ones(1)

	demod_uncertainty_reduction = numpy.sqrt(sum(pow(demod_filt, 2.0)))

	# In the limit of large N, a median reduces uncertainty by sqrt(pi/(2N)),
	# so pretend it's a filter where each coefficient equals sqrt(pi/2) / N.
	median_filt = numpy.ones(median_samples) / median_samples * numpy.sqrt(numpy.pi / 2.0)
	if median_samples < 1:
		median_filt = numpy.ones(1)

	avg_filt = numpy.ones(avg_samples) / avg_samples
	if avg_samples < 1:
		avg_filt = numpy.ones(1)

	effective_filt = numpy.convolve(numpy.convolve(demod_filt, median_filt), avg_filt)
	uncertainty_reduction = numpy.sqrt(sum(pow(effective_filt, 2.0)))

	return pipeparts.mkaudioamplify(pipeline, head, uncertainty_reduction / demod_uncertainty_reduction)

def compute_calline_uncertainty(pipeline, coh_unc, coh_samples, demod_samples, median_samples, avg_samples):
	#
	# The coherence uncertainties may not be equal to the
	# uncertainties in the line ratios after the low-pass filtering
	# (kaiser window), running median, and running mean.
	#

	# I assume that the uncertainty computed from coherence assumes
	# that a mean was used.
	assumed_unc_reduction = 1.0 / numpy.sqrt(coh_samples)

	# Represent each element as a filter with the same effect on uncertainty
	demod_filt = fir.kaiser(demod_samples, 3 * numpy.pi)
	demod_filt /= numpy.sum(demod_filt)
	if demod_samples < 1:
		demod_filt = numpy.ones(1)

	# In the limit of large N, a median reduces uncertainty by sqrt(pi/(2N)),
	# so pretend it's a filter where each coefficient equals sqrt(pi/2) / N.
	median_filt = numpy.ones(median_samples) / median_samples * numpy.sqrt(numpy.pi / 2.0)
	if median_samples < 1:
		median_filt = numpy.ones(1)

	avg_filt = numpy.ones(avg_samples) / avg_samples
	if avg_samples < 1:
		avg_filt = numpy.ones(1)

	effective_filt = numpy.convolve(numpy.convolve(demod_filt, median_filt), avg_filt)
	uncertainty_reduction = numpy.sqrt(sum(pow(effective_filt, 2.0)))

	return pipeparts.mkaudioamplify(pipeline, coh_unc, uncertainty_reduction / assumed_unc_reduction)

def compute_act_stage_uncertainty(pipeline, pcaly_line1_coh, sus_line_coh, coherence_samples, integration_samples, median_smoothing_samples, factors_average_samples, coherence_unc_threshold):
	pcaly_line1_coh_clipped = mkinsertgap(pipeline, pcaly_line1_coh, bad_data_intervals = [0, coherence_unc_threshold], replace_value = coherence_unc_threshold, insert_gap = False)
	sus_line_coh_clipped = mkinsertgap(pipeline, sus_line_coh, bad_data_intervals = [0, coherence_unc_threshold], replace_value = coherence_unc_threshold, insert_gap = False)
	pcaly_line1_unc = compute_calline_uncertainty(pipeline, pcaly_line1_coh_clipped, coherence_samples, integration_samples, median_smoothing_samples, factors_average_samples)
	sus_line_unc = compute_calline_uncertainty(pipeline, sus_line_coh_clipped, coherence_samples, integration_samples, median_smoothing_samples, factors_average_samples)
	return mkpow(pipeline, mkadder(pipeline, list_srcs(pipeline, mkpow(pipeline, pcaly_line1_unc, exponent = 2.0), mkpow(pipeline, sus_line_unc, exponent = 2.0))), exponent = 0.5)

def compute_S_c_uncertainty_from_filters_file(pipeline, EP6_real, EP6_imag, EP7_real, EP7_imag, opt_gain_fcc_line_freq, X2, pcaly_line2_coh, EP8_real, EP8_imag, ktst, tau_tst, apply_complex_kappatst, ktst_unc, EP18_real, EP18_imag, kpum, tau_pum, apply_complex_kappapum, kpum_unc, EP19_real, EP19_imag, kuim, tau_uim, apply_complex_kappauim, kuim_unc, coherence_samples, integration_samples, median_smoothing_samples, factors_average_samples, coherence_unc_threshold):

	#
	# S_c = (1 / EP6) * (X2 - EP7 * (ktst * EP8 + kpum * EP18 + kuim * EP19))^(-1)
	#

	EP7_mag = pow(EP7_real * EP7_real + EP7_imag * EP7_imag, 0.5)
	EP8_mag = pow(EP8_real * EP8_real + EP8_imag * EP8_imag, 0.5)
	EP18_mag = pow(EP18_real * EP18_real + EP18_imag * EP18_imag, 0.5)
	EP19_mag = pow(EP19_real * EP19_real + EP19_imag * EP19_imag, 0.5)

	X2 = pipeparts.mktee(pipeline, X2)
	pcaly_line2_coh_clipped = mkinsertgap(pipeline, pcaly_line2_coh, bad_data_intervals = [0, coherence_unc_threshold], replace_value = coherence_unc_threshold, insert_gap = False)
	X2_unc = compute_calline_uncertainty(pipeline, pcaly_line2_coh_clipped, coherence_samples, integration_samples, median_smoothing_samples, factors_average_samples)
	X2_unc_abs = mkmultiplier(pipeline, list_srcs(pipeline, X2_unc, pipeparts.mkgeneric(pipeline, X2, "cabs")))

	A_tst = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, ktst, matrix = [[EP8_real, EP8_imag]]))
	if apply_complex_kappatst:
		i_omega_tau_tst = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, tau_tst, matrix = [[0, 2 * numpy.pi * opt_gain_fcc_line_freq]]))
		exp_i_omega_tau_tst = pipeparts.mkgeneric(pipeline, i_omega_tau_tst, "cexp")
		A_tst = mkmultiplier(pipeline, list_srcs(pipeline, A_tst, exp_i_omega_tau_tst))
	A_tst_unc_abs = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, ktst, ktst_unc)), EP8_mag)
	A_pum = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, kpum, matrix = [[EP18_real, EP18_imag]]))
	if apply_complex_kappapum:
		i_omega_tau_pum = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, tau_pum, matrix = [[0, 2 * numpy.pi * opt_gain_fcc_line_freq]]))
		exp_i_omega_tau_pum = pipeparts.mkgeneric(pipeline, i_omega_tau_pum, "cexp")
		A_pum = mkmultiplier(pipeline, list_srcs(pipeline, A_pum, exp_i_omega_tau_pum))
	A_pum_unc_abs = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, kpum, kpum_unc)), EP18_mag)
	A_uim = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, kuim, matrix = [[EP19_real, EP19_imag]]))
	if apply_complex_kappauim:
		i_omega_tau_uim = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, tau_uim, matrix = [[0, 2 * numpy.pi * opt_gain_fcc_line_freq]]))
		exp_i_omega_tau_uim = pipeparts.mkgeneric(pipeline, i_omega_tau_uim, "cexp")
		A_uim = mkmultiplier(pipeline, list_srcs(pipeline, A_uim, exp_i_omega_tau_uim))
	A_uim_unc_abs = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, kuim, kuim_unc)), EP19_mag)
	A = pipeparts.mktee(pipeline, mkadder(pipeline, list_srcs(pipeline, A_tst, A_pum, A_uim)))
	A_unc_abs = mkpow(pipeline, mkadder(pipeline, list_srcs(pipeline, mkpow(pipeline, A_tst_unc_abs, exponent = 2.0), mkpow(pipeline, A_pum_unc_abs, exponent = 2.0), mkpow(pipeline, A_uim_unc_abs, exponent = 2.0))), exponent = 0.5)
	minus_DA = complex_audioamplify(pipeline, A, -EP7_real, -EP7_imag)
	DA_unc_abs = pipeparts.mkaudioamplify(pipeline, A_unc_abs, EP7_mag)
	X2_minus_DA = pipeparts.mktee(pipeline, mkadder(pipeline, list_srcs(pipeline, X2, minus_DA)))
	X2_minus_DA_mag = pipeparts.mkgeneric(pipeline, X2_minus_DA, "cabs")
	X2_minus_DA_unc_abs = mkpow(pipeline, mkadder(pipeline, list_srcs(pipeline, mkpow(pipeline, X2_unc_abs, exponent = 2.0), mkpow(pipeline, DA_unc_abs, exponent = 2.0))), exponent = 0.5)
	S_c_unc = pipeparts.mktee(pipeline, complex_division(pipeline, X2_minus_DA_unc_abs, X2_minus_DA_mag))
	S_c = pipeparts.mktee(pipeline, complex_inverse(pipeline, complex_audioamplify(pipeline, X2_minus_DA, EP6_real, EP6_imag)))
	S_c_real = pipeparts.mkgeneric(pipeline, S_c, "creal")
	S_c_imag = pipeparts.mkgeneric(pipeline, S_c, "cimag")
	S_c_product = pipeparts.mkgeneric(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, S_c_real, S_c_imag)), "cabs")
	S_c_square_modulus = pipeparts.mkpow(pipeline, pipeparts.mkgeneric(pipeline, S_c, "cabs"), exponent = 2.0)
	S_c_square_modulus_over_S_c_product = complex_division(pipeline, S_c_square_modulus, S_c_product)
	fcc_unc = mkmultiplier(pipeline, list_srcs(pipeline, S_c_square_modulus_over_S_c_product, S_c_unc))

	return S_c, S_c_unc, fcc_unc

def compute_S_c_uncertainty(pipeline, EP6, EP7, opt_gain_fcc_line_freq, X2, pcaly_line2_coh, EP8, ktst, tau_tst, apply_complex_kappatst, ktst_unc, EP18, kpum, tau_pum, apply_complex_kappapum, kpum_unc, EP19, kuim, tau_uim, apply_complex_kappauim, kuim_unc, coherence_samples, integration_samples, median_smoothing_samples, factors_average_samples, coherence_unc_threshold):

	#
	# S_c = (1 / EP6) * (X2 - EP7 * (ktst * EP8 + kpum * EP18 + kuim * EP19))^(-1)
	#

	EP7_mag = pipeparts.mkgeneric(pipeline, EP7, "cabs")
	EP8_mag = pipeparts.mkgeneric(pipeline, EP8, "cabs")
	EP18_mag = pipeparts.mkgeneric(pipeline, EP18, "cabs")
	EP19_mag = pipeparts.mkgeneric(pipeline, EP19, "cabs")

	X2 = pipeparts.mktee(pipeline, X2)
	pcaly_line2_coh_clipped = mkinsertgap(pipeline, pcaly_line2_coh, bad_data_intervals = [0, coherence_unc_threshold], replace_value = coherence_unc_threshold, insert_gap = False)
	X2_unc = compute_calline_uncertainty(pipeline, pcaly_line2_coh_clipped, coherence_samples, integration_samples, median_smoothing_samples, factors_average_samples)
	X2_unc_abs = mkmultiplier(pipeline, list_srcs(pipeline, X2_unc, pipeparts.mkgeneric(pipeline, X2, "cabs")))

	complex_ktst = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatriximxer(pipeline, ktst, matrix = [[1.0, 0.0]]))
	A_tst = mkmultiplier(pipeline, list_srcs(pipeline, complex_ktst, EP8))
	if apply_complex_kappatst:
		i_omega_tau_tst = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, tau_tst, matrix = [[0, 2 * numpy.pi * opt_gain_fcc_line_freq]]))
		exp_i_omega_tau_tst = pipeparts.mkgeneric(pipeline, i_omega_tau_tst, "cexp")
		A_tst = mkmultiplier(pipeline, list_srcs(pipeline, A_tst, exp_i_omega_tau_tst))
	A_tst_unc_abs = mkmultiplier(pipeline, list_srcs(pipeline, ktst, ktst_unc, EP8_mag))
	complex_kpum = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatriximxer(pipeline, kpum, matrix = [[1.0, 0.0]]))
	A_pum = mkmultiplier(pipeline, list_srcs(pipeline, complex_kpum, EP18))
	if apply_complex_kappapum:
		i_omega_tau_pum = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, tau_pum, matrix = [[0, 2 * numpy.pi * opt_gain_fcc_line_freq]]))
		exp_i_omega_tau_pum = pipeparts.mkgeneric(pipeline, i_omega_tau_pum, "cexp")
		A_pum = mkmultiplier(pipeline, list_srcs(pipeline, A_pum, exp_i_omega_tau_pum))
	A_pum_unc_abs = mkmultiplier(pipeline, list_srcs(pipeline, kpum, kpum_unc, EP18_mag))
	complex_kuim = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatriximxer(pipeline, kuim, matrix = [[1.0, 0.0]]))
	A_uim = mkmultiplier(pipeline, list_srcs(pipeline, complex_kuim, EP19))
	if apply_complex_kappauim:
		i_omega_tau_uim = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, tau_uim, matrix = [[0, 2 * numpy.pi * opt_gain_fcc_line_freq]]))
		exp_i_omega_tau_uim = pipeparts.mkgeneric(pipeline, i_omega_tau_uim, "cexp")
		A_uim = mkmultiplier(pipeline, list_srcs(pipeline, A_uim, exp_i_omega_tau_uim))
	A_uim_unc_abs = mkmultiplier(pipeline, list_srcs(pipeline, kuim, kuim_unc, EP19_mag))
	A = pipeparts.mktee(pipeline, mkadder(pipeline, list_srcs(pipeline, A_tst, A_pum, A_uim)))
	A_unc_abs = mkpow(pipeline, mkadder(pipeline, list_srcs(pipeline, mkpow(pipeline, A_tst_unc_abs, exponent = 2.0), mkpow(pipeline, A_pum_unc_abs, exponent = 2.0), mkpow(pipeline, A_uim_unc_abs, exponent = 2.0))), exponent = 0.5)
	minus_DA = complex_audioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, A, EP7)), -1.0, 0.0)
	DA_unc_abs = mkmultiplier(pipeline, list_srcs(pipeline, A_unc_abs, EP7_mag))
	X2_minus_DA = pipeparts.mktee(pipeline, mkadder(pipeline, list_srcs(pipeline, X2, minus_DA)))
	X2_minus_DA_mag = pipeparts.mkgeneric(pipeline, X2_minus_DA, "cabs")
	X2_minus_DA_unc_abs = mkpow(pipeline, mkadder(pipeline, list_srcs(pipeline, mkpow(pipeline, X2_unc_abs, exponent = 2.0), mkpow(pipeline, DA_unc_abs, exponent = 2.0))), exponent = 0.5)
	S_c_unc = pipeparts.mktee(pipeline, complex_division(pipeline, X2_minus_DA_unc_abs, X2_minus_DA_mag))
	S_c = pipeparts.mktee(pipeline, complex_inverse(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, X2_minus_DA, EP6))))
	S_c_real = pipeparts.mkgeneric(pipeline, S_c, "creal")
	S_c_imag = pipeparts.mkgeneric(pipeline, S_c, "cimag")
	S_c_product = pipeparts.mkgeneric(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, S_c_real, S_c_imag)), "cabs")
	S_c_square_modulus = pipeparts.mkpow(pipeline, pipeparts.mkgeneric(pipeline, S_c, "cabs"), exponent = 2.0)
	S_c_square_modulus_over_S_c_product = complex_division(pipeline, S_c_square_modulus, S_c_product)
	fcc_unc = mkmultiplier(pipeline, list_srcs(pipeline, S_c_square_modulus_over_S_c_product, S_c_unc))

	return S_c, S_c_unc, fcc_unc

def compute_SRC_uncertainty_from_filters_file(pipeline, EP11_real, EP11_imag, kc, kc_unc, fcc, fcc_unc, EP12_real, EP12_imag, act_pcal_line_freq, X1, pcaly_line1_coh, EP13_real, EP13_imag, ktst, tau_tst, apply_complex_kappatst, ktst_unc, EP20_real, EP20_imag, kpum, tau_pum, apply_complex_kappapum, kpum_unc, EP21_real, EP21_imag, kuim, tau_uim, apply_complex_kappauim, kuim_unc, coherence_samples, integration_samples, median_smoothing_samples, factors_average_samples, coherence_unc_threshold):

	#
	# S_s^{-1} = ((EP11 * kc) / (1 + i * f_src / f_cc)) * (pcalfpcal4 / derrfpcal4 - EP12 * (ktst * EP13 + kpum * EP20 + kuim * EP21))
	#

	EP12_mag = pow(EP12_real * EP12_real + EP12_imag * EP12_imag, 0.5)
	EP13_mag = pow(EP13_real * EP13_real + EP13_imag * EP13_imag, 0.5)
	EP20_mag = pow(EP20_real * EP20_real + EP20_imag * EP20_imag, 0.5)
	EP21_mag = pow(EP21_real * EP21_real + EP21_imag * EP21_imag, 0.5)

	X1 = pipeparts.mktee(pipeline, X1)
	fcc = pipeparts.mktee(pipeline, fcc)
	pcaly_line1_coh_clipped = mkinsertgap(pipeline, pcaly_line1_coh, bad_data_intervals = [0, coherence_unc_threshold], replace_value = coherence_unc_threshold, insert_gap = False)
	X1_unc = compute_calline_uncertainty(pipeline, pcaly_line1_coh_clipped, coherence_samples, integration_samples, median_smoothing_samples, factors_average_samples)
	X1_unc_abs = mkmultiplier(pipeline, list_srcs(pipeline, X1_unc, pipeparts.mkgeneric(pipeline, X1, "cabs")))

	A_tst = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, ktst, matrix = [[EP13_real, EP13_imag]]))
	if apply_complex_kappatst:
		i_omega_tau_tst = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, tau_tst, matrix = [[0, 2 * numpy.pi * act_pcal_line_freq]]))
		exp_i_omega_tau_tst = pipeparts.mkgeneric(pipeline, i_omega_tau_tst, "cexp")
		A_tst = mkmultiplier(pipeline, list_srcs(pipeline, A_tst, exp_i_omega_tau_tst))
	A_tst_unc_abs = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, ktst, ktst_unc)), EP13_mag)
	A_pum = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, kpum, matrix = [[EP20_real, EP20_imag]]))
	if apply_complex_kappapum:
		i_omega_tau_pum = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, tau_pum, matrix = [[0, 2 * numpy.pi * act_pcal_line_freq]]))
		exp_i_omega_tau_pum = pipeparts.mkgeneric(pipeline, i_omega_tau_pum, "cexp")
		A_pum = mkmultiplier(pipeline, list_srcs(pipeline, A_pum, exp_i_omega_tau_pum))
	A_pum_unc_abs = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, kpum, kpum_unc)), EP20_mag)
	A_uim = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, kuim, matrix = [[EP21_real, EP21_imag]]))
	if apply_complex_kappauim:
		i_omega_tau_uim = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, tau_uim, matrix = [[0, 2 * numpy.pi * act_pcal_line_freq]]))
		exp_i_omega_tau_uim = pipeparts.mkgeneric(pipeline, i_omega_tau_uim, "cexp")
		A_uim = mkmultiplier(pipeline, list_srcs(pipeline, A_uim, exp_i_omega_tau_uim))
	A_uim_unc_abs = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, kuim, kuim_unc)), EP21_mag)
	A = pipeparts.mktee(pipeline, mkadder(pipeline, list_srcs(pipeline, A_tst, A_pum, A_uim)))
	A_unc_abs = mkpow(pipeline, mkadder(pipeline, list_srcs(pipeline, mkpow(pipeline, A_tst_unc_abs, exponent = 2.0), mkpow(pipeline, A_pum_unc_abs, exponent = 2.0), mkpow(pipeline, A_uim_unc_abs, exponent = 2.0))), exponent = 0.5)
	minus_DA = complex_audioamplify(pipeline, A, -EP12_real, -EP12_imag)
	DA_unc_abs = pipeparts.mkaudioamplify(pipeline, A_unc_abs, EP12_mag)
	X1_minus_DA = pipeparts.mktee(pipeline, mkadder(pipeline, list_srcs(pipeline, X1, minus_DA)))
	X1_minus_DA_mag = pipeparts.mkgeneric(pipeline, X1_minus_DA, "cabs")
	X1_minus_DA_unc_abs = mkpow(pipeline, mkadder(pipeline, list_srcs(pipeline, mkpow(pipeline, X1_unc_abs, exponent = 2.0), mkpow(pipeline, DA_unc_abs, exponent = 2.0))), exponent = 0.5)
	S_s_S_c_inverse_unc_squared = pipeparts.mktee(pipeline, mkpow(pipeline, complex_division(pipeline, X1_minus_DA_unc_abs, X1_minus_DA_mag), exponent = 2.0))
	S_s_S_c_inverse = pipeparts.mktee(pipeline, complex_audioamplify(pipeline, X1_minus_DA, EP11_real, EP11_imag))
	S_s_S_c_inverse_real_squared = pipeparts.mktee(pipeline, mkpow(pipeline, pipeparts.mkgeneric(pipeline, S_s_S_c_inverse, "creal"), exponent = 2.0))
	S_s_S_c_inverse_imag_squared = pipeparts.mktee(pipeline, mkpow(pipeline, pipeparts.mkgeneric(pipeline, S_s_S_c_inverse, "cimag"), exponent = 2.0))

	complex_kc = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, kc, [[1.0, 0.0]]))
	f_over_fcc = complex_inverse(pipeline, pipeparts.mkaudioamplify(pipeline, fcc, 1.0 / act_pcal_line_freq))
	i_f_over_fcc = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, f_over_fcc, [[0.0, 1.0]]))
	one_plus_i_f_over_fcc = pipeparts.mkgeneric(pipeline, i_f_over_fcc, "lal_add_constant", value = 1.0)
	S_c = pipeparts.mktee(pipeline, complex_division(pipeline, complex_kc, one_plus_i_f_over_fcc))
	S_c_real_squared = pipeparts.mktee(pipeline, mkpow(pipeline, pipeparts.mkgeneric(pipeline, S_c, "creal"), exponent = 2.0))
	S_c_imag_squared = pipeparts.mktee(pipeline, mkpow(pipeline, pipeparts.mkgeneric(pipeline, S_c, "cimag"), exponent = 2.0))

	S_s_inv = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, S_s_S_c_inverse, S_c)))
	pipeparts.mkfakesink(pipeline, S_s_inv)

	f_squared_over_fcc_squared = pipeparts.mktee(pipeline, mkpow(pipeline, pipeparts.mkaudioamplify(pipeline, fcc, 1.0 / act_pcal_line_freq), exponent = -2.0))
	fcc_unc_squared = pipeparts.mktee(pipeline, mkpow(pipeline, fcc_unc, exponent = 2.0))
	inv_denominator = mkpow(pipeline, pipeparts.mkgeneric(pipeline, f_squared_over_fcc_squared, "lal_add_constant", value = 1.0), exponent = -2.0)
	numerator = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, fcc_unc_squared, mkpow(pipeline, f_squared_over_fcc_squared, exponent = 2.0))), 2.0)
	S_c_unc_real_squared = pipeparts.mktee(pipeline, mkadder(pipeline, list_srcs(pipeline, mkpow(pipeline, kc_unc, exponent = 2.0), mkmultiplier(pipeline, list_srcs(pipeline, numerator, inv_denominator)))))
	S_c_unc_imag_squared = pipeparts.mktee(pipeline, mkadder(pipeline, list_srcs(pipeline, S_c_unc_real_squared, fcc_unc_squared)))
	S_s_inverse_real_unc_abs_squared = mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, S_c_real_squared, S_s_S_c_inverse_real_squared, mkadder(pipeline, list_srcs(pipeline, S_c_unc_real_squared, S_s_S_c_inverse_unc_squared)))), mkmultiplier(pipeline, list_srcs(pipeline, S_c_imag_squared, S_s_S_c_inverse_imag_squared, mkadder(pipeline, list_srcs(pipeline, S_c_unc_imag_squared, S_s_S_c_inverse_unc_squared))))))
	S_s_inverse_imag_unc_abs_squared = mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, S_c_real_squared, S_s_S_c_inverse_imag_squared, mkadder(pipeline, list_srcs(pipeline, S_c_unc_real_squared, S_s_S_c_inverse_unc_squared)))), mkmultiplier(pipeline, list_srcs(pipeline, S_c_imag_squared, S_s_S_c_inverse_real_squared, mkadder(pipeline, list_srcs(pipeline, S_c_unc_imag_squared, S_s_S_c_inverse_unc_squared))))))
	fs_squared_unc_abs = pipeparts.mkaudioamplify(pipeline, mkpow(pipeline, S_s_inverse_real_unc_abs_squared, exponent = 0.5), act_pcal_line_freq * act_pcal_line_freq)
	fs_over_Q_unc_abs = pipeparts.mkaudioamplify(pipeline, mkpow(pipeline, S_s_inverse_imag_unc_abs_squared, exponent = 0.5), act_pcal_line_freq)

	return S_s_inv, fs_squared_unc_abs, fs_over_Q_unc_abs

def compute_SRC_uncertainty(pipeline, EP11, kc, kc_unc, fcc, fcc_unc, EP12, act_pcal_line_freq, X1, pcaly_line1_coh, EP13, ktst, tau_tst, apply_complex_kappatst, ktst_unc, EP20, kpum, tau_pum, apply_complex_kappapum, kpum_unc, EP21, kuim, tau_uim, apply_complex_kappauim, kuim_unc, coherence_samples, integration_samples, median_smoothing_samples, factors_average_samples, coherence_unc_threshold):

	#
	# S_s^{-1} = ((EP11 * kc) / (1 + i * f_src / f_cc)) * (pcalfpcal4 / derrfpcal4 - EP12 * (ktst * EP13 + kpum * EP20 + kuim * EP21))
	#

	EP12_mag = pipeparts.mkgeneric(pipeline, EP12, "cabs")
	EP13_mag = pipeparts.mkgeneric(pipeline, EP13, "cabs")
	EP20_mag = pipeparts.mkgeneric(pipeline, EP20, "cabs")
	EP21_mag = pipeparts.mkgeneric(pipeline, EP21, "cabs")

	X1 = pipeparts.mktee(pipeline, X1)
	fcc = pipeparts.mktee(pipeline, fcc)
	pcaly_line1_coh_clipped = mkinsertgap(pipeline, pcaly_line1_coh, bad_data_intervals = [0, coherence_unc_threshold], replace_value = coherence_unc_threshold, insert_gap = False)
	X1_unc = compute_calline_uncertainty(pipeline, pcaly_line1_coh_clipped, coherence_samples, integration_samples, median_smoothing_samples, factors_average_samples)
	X1_unc_abs = mkmultiplier(pipeline, list_srcs(pipeline, X1_unc, pipeparts.mkgeneric(pipeline, X1, "cabs")))

	A_tst = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, ktst, matrix = [[1.0, 0.0]])), EP13))
	if apply_complex_kappatst:
		i_omega_tau_tst = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, tau_tst, matrix = [[0, 2 * numpy.pi * act_pcal_line_freq]]))
		exp_i_omega_tau_tst = pipeparts.mkgeneric(pipeline, i_omega_tau_tst, "cexp")
		A_tst = mkmultiplier(pipeline, list_srcs(pipeline, A_tst, exp_i_omega_tau_tst))
	A_tst_unc_abs = mkmultiplier(pipeline, list_srcs(pipeline, ktst, ktst_unc, EP13_mag))
	A_pum = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, kpum, matrix = [[1.0, 0.0]])), EP20))
	if apply_complex_kappapum:
		i_omega_tau_pum = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, tau_pum, matrix = [[0, 2 * numpy.pi * act_pcal_line_freq]]))
		exp_i_omega_tau_pum = pipeparts.mkgeneric(pipeline, i_omega_tau_pum, "cexp")
		A_pum = mkmultiplier(pipeline, list_srcs(pipeline, A_pum, exp_i_omega_tau_pum))
	A_pum_unc_abs = mkmultiplier(pipeline, list_srcs(pipeline, kpum, kpum_unc, EP20_mag))
	A_uim = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, kuim, matrix = [[1.0, 0.0]])), EP21))
	if apply_complex_kappauim:
		i_omega_tau_uim = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, tau_uim, matrix = [[0, 2 * numpy.pi * act_pcal_line_freq]]))
		exp_i_omega_tau_uim = pipeparts.mkgeneric(pipeline, i_omega_tau_uim, "cexp")
		A_uim = mkmultiplier(pipeline, list_srcs(pipeline, A_uim, exp_i_omega_tau_uim))
	A_uim_unc_abs = mkmultiplier(pipeline, list_srcs(pipeline, kuim, kuim_unc, EP21_mag))
	A = pipeparts.mktee(pipeline, mkadder(pipeline, list_srcs(pipeline, A_tst, A_pum, A_uim)))
	A_unc_abs = mkpow(pipeline, mkadder(pipeline, list_srcs(pipeline, mkpow(pipeline, A_tst_unc_abs, exponent = 2.0), mkpow(pipeline, A_pum_unc_abs, exponent = 2.0), mkpow(pipeline, A_uim_unc_abs, exponent = 2.0))), exponent = 0.5)
	minus_DA = complex_audioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, A, EP12)), -1.0, 0.0)
	DA_unc_abs = mkmultiplier(pipeline, list_srcs(pipeline, A_unc_abs, EP12_mag))
	X1_minus_DA = pipeparts.mktee(pipeline, mkadder(pipeline, list_srcs(pipeline, X1, minus_DA)))
	X1_minus_DA_mag = pipeparts.mkgeneric(pipeline, X1_minus_DA, "cabs")
	X1_minus_DA_unc_abs = mkpow(pipeline, mkadder(pipeline, list_srcs(pipeline, mkpow(pipeline, X1_unc_abs, exponent = 2.0), mkpow(pipeline, DA_unc_abs, exponent = 2.0))), exponent = 0.5)
	S_s_S_c_inverse_unc_squared = pipeparts.mktee(pipeline, mkpow(pipeline, complex_division(pipeline, X1_minus_DA_unc_abs, X1_minus_DA_mag), exponent = 2.0))
	S_s_S_c_inverse = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, X1_minus_DA, EP11)))
	S_s_S_c_inverse_real_squared = pipeparts.mktee(pipeline, mkpow(pipeline, pipeparts.mkgeneric(pipeline, S_s_S_c_inverse, "creal"), exponent = 2.0))
	S_s_S_c_inverse_imag_squared = pipeparts.mktee(pipeline, mkpow(pipeline, pipeparts.mkgeneric(pipeline, S_s_S_c_inverse, "cimag"), exponent = 2.0))

	complex_kc = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, kc, [[1.0, 0.0]]))
	f_over_fcc = complex_inverse(pipeline, pipeparts.mkaudioamplify(pipeline, fcc, 1.0 / act_pcal_line_freq))
	i_f_over_fcc = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, f_over_fcc, [[0.0, 1.0]]))
	one_plus_i_f_over_fcc = pipeparts.mkgeneric(pipeline, i_f_over_fcc, "lal_add_constant", value = 1.0)
	S_c = pipeparts.mktee(pipeline, complex_division(pipeline, complex_kc, one_plus_i_f_over_fcc))
	S_c_real_squared = pipeparts.mktee(pipeline, mkpow(pipeline, pipeparts.mkgeneric(pipeline, S_c, "creal"), exponent = 2.0))
	S_c_imag_squared = pipeparts.mktee(pipeline, mkpow(pipeline, pipeparts.mkgeneric(pipeline, S_c, "cimag"), exponent = 2.0))

	S_s_inv = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, S_s_S_c_inverse, S_c)))
	pipeparts.mkfakesink(pipeline, S_s_inv)

	f_squared_over_fcc_squared = pipeparts.mktee(pipeline, mkpow(pipeline, pipeparts.mkaudioamplify(pipeline, fcc, 1.0 / act_pcal_line_freq), exponent = -2.0))
	fcc_unc_squared = pipeparts.mktee(pipeline, mkpow(pipeline, fcc_unc, exponent = 2.0))
	inv_denominator = mkpow(pipeline, pipeparts.mkgeneric(pipeline, f_squared_over_fcc_squared, "lal_add_constant", value = 1.0), exponent = -2.0)
	numerator = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, fcc_unc_squared, mkpow(pipeline, f_squared_over_fcc_squared, exponent = 2.0))), 2.0)
	S_c_unc_real_squared = pipeparts.mktee(pipeline, mkadder(pipeline, list_srcs(pipeline, mkpow(pipeline, kc_unc, exponent = 2.0), mkmultiplier(pipeline, list_srcs(pipeline, numerator, inv_denominator)))))
	S_c_unc_imag_squared = pipeparts.mktee(pipeline, mkadder(pipeline, list_srcs(pipeline, S_c_unc_real_squared, fcc_unc_squared)))
	S_s_inverse_real_unc_abs_squared = mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, S_c_real_squared, S_s_S_c_inverse_real_squared, mkadder(pipeline, list_srcs(pipeline, S_c_unc_real_squared, S_s_S_c_inverse_unc_squared)))), mkmultiplier(pipeline, list_srcs(pipeline, S_c_imag_squared, S_s_S_c_inverse_imag_squared, mkadder(pipeline, list_srcs(pipeline, S_c_unc_imag_squared, S_s_S_c_inverse_unc_squared))))))
	S_s_inverse_imag_unc_abs_squared = mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, S_c_real_squared, S_s_S_c_inverse_imag_squared, mkadder(pipeline, list_srcs(pipeline, S_c_unc_real_squared, S_s_S_c_inverse_unc_squared)))), mkmultiplier(pipeline, list_srcs(pipeline, S_c_imag_squared, S_s_S_c_inverse_real_squared, mkadder(pipeline, list_srcs(pipeline, S_c_unc_imag_squared, S_s_S_c_inverse_unc_squared))))))
	fs_squared_unc_abs = pipeparts.mkaudioamplify(pipeline, mkpow(pipeline, S_s_inverse_real_unc_abs_squared, exponent = 0.5), act_pcal_line_freq * act_pcal_line_freq)
	fs_over_Q_unc_abs = pipeparts.mkaudioamplify(pipeline, mkpow(pipeline, S_s_inverse_imag_unc_abs_squared, exponent = 0.5), act_pcal_line_freq)

	return S_s_inv, fs_squared_unc_abs, fs_over_Q_unc_abs

def find_injection_ratios_from_model(filters, ktst = 1.0, tau_tst = 0.0, kpum = 1.0, tau_pum = 0.0, kuim = 1.0, tau_uim = 0.0, kc = 1.0, fcc = None, fs_squared = None, Qinv = None):
	#
	# Find the model-predicted X's of eqs. 5.2.22 and 5.2.23 in P1900052, given
	# a filters file and values for the TDCFs.  Useful mainly for testing.
	#

	f1 = float(filters['ka_pcal_line_freq'])
	f2 = float(filters['kc_pcal_line_freq'])
	fT = float(filters['ktst_esd_line_freq'])
	fP = float(filters['pum_act_line_freq'])
	fU = float(filters['uim_act_line_freq'])

	Cres1 = float(filters['EP11_real']) + 1j * float(filters['EP11_imag'])
	CresDAT1 = float(filters['EP25_real']) + 1j * float(filters['EP25_imag'])
	CresDAP1 = float(filters['EP26_real']) + 1j * float(filters['EP26_imag'])
	CresDAU1 = float(filters['EP27_real']) + 1j * float(filters['EP27_imag'])

	Cres2 = float(filters['EP6_real']) + 1j * float(filters['EP6_imag'])
	CresDAT2 = float(filters['EP28_real']) + 1j * float(filters['EP28_imag'])
	CresDAP2 = float(filters['EP29_real']) + 1j * float(filters['EP29_imag'])
	CresDAU2 = float(filters['EP30_real']) + 1j * float(filters['EP30_imag'])

	CresAT0T = float(filters['EP31_real']) + 1j * float(filters['EP31_imag'])
	CresDATT = float(filters['EP32_real']) + 1j * float(filters['EP32_imag'])
	CresDAPT = float(filters['EP33_real']) + 1j * float(filters['EP33_imag'])
	CresDAUT = float(filters['EP34_real']) + 1j * float(filters['EP34_imag'])

	CresAP0P = float(filters['EP35_real']) + 1j * float(filters['EP35_imag'])
	CresDATP = float(filters['EP36_real']) + 1j * float(filters['EP36_imag'])
	CresDAPP = float(filters['EP37_real']) + 1j * float(filters['EP37_imag'])
	CresDAUP = float(filters['EP38_real']) + 1j * float(filters['EP38_imag'])

	CresAU0U = float(filters['EP39_real']) + 1j * float(filters['EP39_imag'])
	CresDATU = float(filters['EP40_real']) + 1j * float(filters['EP40_imag'])
	CresDAPU = float(filters['EP41_real']) + 1j * float(filters['EP41_imag'])
	CresDAUU = float(filters['EP42_real']) + 1j * float(filters['EP42_imag'])

	fcc_model = float(filters['fcc'])
	if fcc is None:
		fcc = fcc_model
	fs_squared_model = float(filters['fs_squared'])
	if fs_squared is None:
		fs_squared = fs_squared_model
	Qinv_model = 1.0 / float(filters['srcQ'])
	if Qinv is None:
		Qinv = Qinv_model

	CresDAT1 *= ktst * numpy.exp(2.0 * numpy.pi * 1j * f1 * tau_tst)
	CresDAP1 *= kpum * numpy.exp(2.0 * numpy.pi * 1j * f1 * tau_pum)
	CresDAU1 *= kuim * numpy.exp(2.0 * numpy.pi * 1j * f1 * tau_uim)

	CresDAT2 *= ktst * numpy.exp(2.0 * numpy.pi * 1j * f2 * tau_tst)
	CresDAP2 *= kpum * numpy.exp(2.0 * numpy.pi * 1j * f2 * tau_pum)
	CresDAU2 *= kuim * numpy.exp(2.0 * numpy.pi * 1j * f2 * tau_uim)

	CresAT0T *= ktst * numpy.exp(2.0 * numpy.pi * 1j * fT * tau_tst)
	CresDATT *= ktst * numpy.exp(2.0 * numpy.pi * 1j * fT * tau_tst)
	CresDAPT *= kpum * numpy.exp(2.0 * numpy.pi * 1j * fT * tau_pum)
	CresDAUT *= kuim * numpy.exp(2.0 * numpy.pi * 1j * fT * tau_uim)

	CresAP0P *= kpum * numpy.exp(2.0 * numpy.pi * 1j * fP * tau_pum)
	CresDATP *= ktst * numpy.exp(2.0 * numpy.pi * 1j * fP * tau_tst)
	CresDAPP *= kpum * numpy.exp(2.0 * numpy.pi * 1j * fP * tau_pum)
	CresDAUP *= kuim * numpy.exp(2.0 * numpy.pi * 1j * fP * tau_uim)

	CresAU0U *= kuim * numpy.exp(2.0 * numpy.pi * 1j * fU * tau_uim)
	CresDATU *= ktst * numpy.exp(2.0 * numpy.pi * 1j * fU * tau_tst)
	CresDAPU *= kpum * numpy.exp(2.0 * numpy.pi * 1j * fU * tau_pum)
	CresDAUU *= kuim * numpy.exp(2.0 * numpy.pi * 1j * fU * tau_uim)

	CresX1 = CresDAT1 + CresDAP1 + CresDAU1 + ((1.0 + 1j * f1 / fcc) / kc) * ((f1 * f1 + fs_squared - 1j * f1 * numpy.sqrt(abs(fs_squared)) * Qinv) / (f1 * f1))
	CresX2 = CresDAT2 + CresDAP2 + CresDAU2 + ((1.0 + 1j * f2 / fcc) / kc) * ((f2 * f2 + fs_squared - 1j * f2 * numpy.sqrt(abs(fs_squared)) * Qinv) / (f2 * f2))
	CresAT0XT = CresDATT + CresDAPT + CresDAUT + ((1.0 + 1j * fT / fcc) / kc) * ((fT * fT + fs_squared - 1j * fT * numpy.sqrt(abs(fs_squared)) * Qinv) / (fT * fT))
	CresAP0XP = CresDATP + CresDAPP + CresDAUP + ((1.0 + 1j * fP / fcc) / kc) * ((fP * fP + fs_squared - 1j * fP * numpy.sqrt(abs(fs_squared)) * Qinv) / (fP * fP))
	CresAU0XU = CresDATU + CresDAPU + CresDAUU + ((1.0 + 1j * fU / fcc) / kc) * ((fU * fU + fs_squared - 1j * fU * numpy.sqrt(abs(fs_squared)) * Qinv) / (fU * fU))

	X1 = CresX1 / Cres1
	X2 = CresX2 / Cres2
	XT = CresAT0XT / CresAT0T
	XP = CresAP0XP / CresAP0P
	XU = CresAU0XU / CresAU0U

	return X1, X2, XT, XP, XU

def update_property_simple(prop_maker, arg, prop_taker, maker_prop_name, taker_prop_name, prefactor):
	prop = prop_maker.get_property(maker_prop_name)
	prop_taker.set_property(taker_prop_name, prefactor * prop)

def update_timestamped_property(prop_maker, arg, prop_taker, maker_prop_name, taker_prop_name, prefactor):
	prop = prop_maker.get_property(maker_prop_name)
	cs = GstController.InterpolationControlSource.new()
	binding = GstController.DirectControlBinding.new_absolute(prop_taker, taker_prop_name, cs)
	prop_taker.add_control_binding(binding)
	cs.set_property('mode', GstController.InterpolationMode.NONE) # no interpolation
	cs.set(int(prop[0] * Gst.SECOND), prefactor * prop[1])

def update_filter(filter_maker, arg, filter_taker, maker_prop_name, taker_prop_name):
	firfilter = filter_maker.get_property(maker_prop_name)[::-1]
	filter_taker.set_property(taker_prop_name, firfilter)

def update_filters(filter_maker, arg, filter_taker, maker_prop_name, taker_prop_name, filter_number):
	firfilter = filter_maker.get_property(maker_prop_name)[filter_number][::-1]
	filter_taker.set_property(taker_prop_name, firfilter)

def clean_data(pipeline, signal, signal_rate, witnesses, witness_rate, fft_length, fft_overlap, num_ffts, min_ffts, update_samples, fir_length, frequency_resolution, filter_taper_length, use_median = False, parallel_mode = False, notch_frequencies = [], high_pass = 15.0, noisesub_gate_bit = None, delay_time = 0.0, critical_lock_loss_time = 0, fft_window_type = 'dpss', fir_window_type = 'dpss', filename = None):

	#
	# Use witness channels that monitor the environment to remove environmental noise
	# from a signal of interest.  This function accounts for potential correlation
	# between witness channels.
	#

	signal_tee = pipeparts.mktee(pipeline, signal)
	witnesses = list(witnesses)
	witness_tees = []
	for i in range(0, len(witnesses)):
		witnesses[i] = mkresample(pipeline, witnesses[i], 4, False, witness_rate)
		witness_tees.append(pipeparts.mktee(pipeline, witnesses[i]))

	resampled_signal = mkresample(pipeline, signal_tee, 4, False, witness_rate)
	transfer_functions = mkinterleave(pipeline, numpy.insert(witness_tees, 0, resampled_signal, axis = 0))
	if noisesub_gate_bit is not None:
		transfer_functions = mkgate(pipeline, transfer_functions, noisesub_gate_bit, 1)
	transfer_functions = mktransferfunction(pipeline, transfer_functions, fft_length = fft_length, fft_overlap = fft_overlap, num_ffts = num_ffts, min_ffts = min_ffts, update_samples = update_samples, make_fir_filters = -1, fir_length = fir_length, frequency_resolution = frequency_resolution, high_pass = high_pass / 2.0, update_after_gap = True, use_median = use_median, parallel_mode = parallel_mode, notch_frequencies = notch_frequencies, use_first_after_gap = critical_lock_loss_time * witness_rate, update_delay_samples = int(delay_time * witness_rate), fir_timeshift = 0, fft_window_type = fft_window_type, fir_window_type = fir_window_type, filename = filename)
	signal_minus_noise = [signal_tee]
	for i in range(0, len(witnesses)):
		if parallel_mode:
			minus_noise = pipeparts.mkgeneric(pipeline, mkqueue(pipeline, highpass(pipeline, witness_tees[i], witness_rate, fcut = high_pass, freq_res = high_pass / 3.0)), "lal_tdwhiten", kernel = numpy.zeros(fir_length), latency = fir_length // 2, taper_length = filter_taper_length, kernel_endtime = 0)
			transfer_functions.connect("notify::fir-filters", update_filters, minus_noise, "fir_filters", "kernel", i)
			transfer_functions.connect("notify::fir-endtime", update_property_simple, minus_noise, "fir_endtime", "kernel_endtime", 1)
		else:
			minus_noise = pipeparts.mkgeneric(pipeline, highpass(pipeline, witness_tees[i], witness_rate, fcut = high_pass, freq_res = high_pass / 3.0), "lal_tdwhiten", kernel = numpy.zeros(fir_length), latency = fir_length // 2, taper_length = filter_taper_length)
			transfer_functions.connect("notify::fir-filters", update_filters, minus_noise, "fir_filters", "kernel", i)
		signal_minus_noise.append(mkresample(pipeline, minus_noise, 4, False, signal_rate))

	return mkadder(pipeline, tuple(signal_minus_noise))



