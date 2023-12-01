#
# Copyright 2008,2009 Free Software Foundation, Inc.
#
# This application is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This application is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

# The presence of this file turns this directory into a Python package

'''
This is the GNU Radio OFDM_TOOLS module. Place your Python package
description here (python/__init__.py).
'''

# ----------------------------------------------------------------
# Temporary workaround for ticket:181 (swig+python problem)
import sys
_RTLD_GLOBAL = 0
try:
    from dl import RTLD_GLOBAL as _RTLD_GLOBAL
except ImportError:
    try:
	from DLFCN import RTLD_GLOBAL as _RTLD_GLOBAL
    except ImportError:
	pass

if _RTLD_GLOBAL != 0:
    _dlopenflags = sys.getdlopenflags()
    sys.setdlopenflags(_dlopenflags|_RTLD_GLOBAL)
# ----------------------------------------------------------------


# import swig generated symbols into the ofdm_tools namespace
#from ofdm_tools_swig import *

# import any pure python here

from papr_sink import papr_sink
from ofdm_radio_hier import ofdm_radio_hier
from payload_source import payload_source
from payload_sink import payload_sink
from spectrum_sensor import spectrum_sensor
from cognitive_engine_mac import cognitive_engine_mac
from sync_radio_hier import sync_radio_hier
from ascii_plot import ascii_plot
from ais_decoder import ais_decoder
from fosphor_main import fosphor_main
from spectrum_logger import spectrum_logger
from psd_logger import psd_logger
from spectrum_sensor_v1 import spectrum_sensor_v1
from flanck_detector import flanck_detector
from ofdm_tx_rx_hier import ofdm_tx_rx_hier
from spectrum_sensor_v2 import spectrum_sensor_v2
from message_pdu import message_pdu
from coherence_detector import coherence_detector
from payload_source_pdu import payload_source_pdu
from payload_sink_pdu import payload_sink_pdu
from chat_blocks import chat_sender, chat_receiver
from ascii_gnuplot import ascii_gnuplot
from multichannel_scanner import multichannel_scanner
from local_worker import local_worker
from remote_client import remote_client


from ascii_plot import ascii_plotter, ascii_bars
from remote_client_qt import remote_client_qt
from spectrum_sweeper import spectrum_sweeper
from clipper import clipper
from uart_serial import uart_serial
from dump1090_interface import dump1090_interface

import ofdm_txrx_modules
import ofdm_cr_tools

#

# ----------------------------------------------------------------
# Tail of workaround
if _RTLD_GLOBAL != 0:
    sys.setdlopenflags(_dlopenflags)      # Restore original flags
# ----------------------------------------------------------------
