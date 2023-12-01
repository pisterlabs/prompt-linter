# This Device Class is published under the terms of the MIT License.
# Required Third Party Libraries, which are included in the Device Class
# package for convenience purposes, may have a different license. You can
# find those in the corresponding folders or contact the maintainer.
#
# MIT License
#
# Copyright (c) 2022-2023 SweepMe! GmbH (sweep-me.net )
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time

from pysweepme.EmptyDeviceClass import EmptyDevice  # Class comes with SweepMe!
# from pysweepme.ErrorMessage import error
from pysweepme.FolderManager import addFolderToPATH

addFolderToPATH()
from coherent_errors import LASER_ERROR_CODES


class ChameleonError(Exception):
    """wrapped generic exception"""


class Device(EmptyDevice):

    description = """
                <h3>Coherent Chameleon Laser driver</h3>
                <p>---Switch driver for the Chameleon series, developped for Ultra 2 variant
                ---</p>
                <p><FONT COLOR="#ff0000"> <b>Safety Warning</b>: the user is responsible for the 
                safe operation of the laser and checking that the laser is in a safe state after
                 each SweepMe! run finishes.</p>
                <p><b>Features</b>:</p>
                <ul>{ padding: 5px 0px; }
                <li> Overview: The driver supports wavelength sweeps and shutter control as well as laser
                 alignment mode.<li>
                <li> Tuning: after a new wavelength is requested, the driver will wait until tuning is
                 complete.</li>
                <li style="margin-left:2em"> If tuning times out, an error will be raised. </li>
                <li> Modelocking: If tuning is successful, modelocking is attempted 
                 (with user settable timeout) 
                <li style="margin-left:2em"> If modelock is times out a warning is raised. Modelock
                 status is saved as a variable</li>
                </ul>
                <p><b>User options</b>:</p>
                <ul>
                <li>Close shutter while tuning: automatically close the shutter while changing
                 wavelength and waiting for tuning/modelock to complete </li>
                <li>Open shutter at start: automatically open shutter when entering sequencer
                 branches containing the laser</li>
                <li>Close shutter at start: automatically close shutter when exiting sequencer
                 branches containing the laser</li>
                <li>Alignment mode: set the laser to alignment mode. <b> in this mode the laser
                 cannot change wavelength. Alignment mode will be automatically exited
                  and re-entered during a wavelength change. In this mode Close shutter while
                  tuning will be set to true<\b> </li>
                </ul>
                <p>&nbsp;</p>
                """
    actions = ["close_shutter", "open_shutter", "set_laser_on", "set_laser_on_standby"]

    def __init__(self):
        EmptyDevice.__init__(self)

        self.shortname = "Chameleon"  # short name will be shown in the sequencer
        self.variables = ["Wavelength", "Shutter open", "Power", "Mode locked"]
        self.units = ["nm", "", "mW", ""]
        self.plottype = [True, True, True, False]
        self.savetype = [True, True, True, True]

        # serial comms
        self.port_manager = True
        self.port_types = ["COM"]
        # https://wiki.sweep-me.net/wiki/Port_manager#Port_properties
        self.port_properties = {"timeout": 10, "baudrate": 19200, "EOL": '\r\n'}

        # tracking & init
        self._is_shutter_open = False
        self.wln = 690
        self._is_mode_locked = None
        # user options
        self.start_wln = self.wln
        self._open_shutter_on_start = True
        self._close_shutter_at_end = True
        self._auto_shutter = True

    def set_GUIparameter(self):

        # wlns for the model are with 1nm resolution so can use int
        GUIparameter = {
            # "Start up wavelength": 690,
            "Close shutter while tuning": True,
            "Open shutter at start": True,
            "Close shutter at end": True,
            "": None,
            "Modelock search timeout in s": 10.0,
            "Alignment mode": False,
            "SweepMode": ["Wavelength in nm"]
        }
        return GUIparameter

    def get_GUIparameter(self, parameter):

        # self.start_wln = int(parameter["Start up wavelength"])
        # shutter control
        self._open_shutter_on_start = parameter["Open shutter at start"]
        self._close_shutter_at_end = parameter["Close shutter at end"]
        self._auto_shutter: bool = parameter["Close shutter while tuning"]
        self.port_string = parameter["Port"]  # auto used by port manager
        self.alignment_mode: bool = parameter["Alignment mode"]
        # for safety reasons: in alignment mode need to exit the alignment
        # during wln changes. This results in a power increase which is dangerous
        if self.alignment_mode:
            self._auto_shutter = True
            print("Warning: auto shutter set to True due to alignment mode selection")
        self.modelock_timeout = float(parameter["Modelock search timeout in s"])
        self.sweepmode: str = parameter["SweepMode"]

    def disconnect(self):
        # called only once at the end of the measurement
        self.set_shutter_state(open_=False)

    def initialize(self):
        """ perform initialisation steps needed only once per entire measurement"""

        # set echo mode off for comms
        self.write_port("ECHO=0")
        # set prompt mode off for comms, which may have been turned on by the chameleonGUI
        self.write_port("PROMPT=0")
        # initial checks
        errors = self.check_errors()
        if errors:
            print("ERRORS: ", errors)
        is_key_ok = self.check_key_turned()
        if not is_key_ok:
            raise ValueError("LASER KEY IS IN OFF STATE")
        is_laser_on = self.get_laser_status()
        if not is_laser_on:
            raise ValueError("LASER IS NOT ON")

        # initialisation
        self._is_shutter_open = self.get_shutter_state()
        # if self.start_wln:
        #     self.set_wavelength(self.start_wln)

    def poweron(self):
        # called if the measurement procedure enters a branch of the sequencer
        # and the module has not been used in the previous branch
        self.set_laser_on()
        # print("laser power on")

    def poweroff(self):
        # called if the measurement procedure leaves a branch of the sequencer
        # and the module is not used in the next branch

        # self.set_laser_on_standby()
        if self._close_shutter_at_end:
            self.set_shutter_state(open_=False)
        # print("laser power off  ")

    def configure(self):
        # called if the measurement procedure enters a branch of the sequencer
        # and the module has not been used in the previous branch
        # print("laser alignment on?: ", self.is_in_alignment_mode())
        self.set_alignment_mode(self.alignment_mode)

        if self._open_shutter_on_start:
            self.set_shutter_state(open_=True)

    def deinitialize(self):
        #     # called if the measurement procedure leaves a branch of the sequencer and
        #     # the module is not used in the next branch
        errors = self.check_errors()
        if errors:
            print("Errors for laser after measurement: ", errors)

    # def reconfigure(self, parameters, keys):
    #     """ 'reconfigure' is called whenever parameters of the GUI change by
    #      using the {...}-parameter system """
    #     print("reconfigure")
    #     print("Parameters:", parameters)
    #     print("Changed keys:", keys)

    #    """ the following functions are called for each measurement point """

    def apply(self):
        """ 'apply' is used to set the new setvalue that is always available as 'self.value'
            and is only called if the setvalue has changed """

        if self._auto_shutter:
            self.close_shutter()

        if self.sweepmode == "Wavelength in nm":
            if self.alignment_mode:
                print("WARNING: EXITING ALIGNMENT MODE TO CHANGE WLN")
                self.set_alignment_mode(False)

            if not self.is_in_alignment_mode():
                self.set_wavelength(self.value)

            if self.alignment_mode:
                self.set_alignment_mode(self.alignment_mode)
        else:
            raise ValueError(f"APPLY CALLED FOR UNSUPPORTED SWEEPMODE {self.sweepmode}")

    def reach(self):
        """ can be added to make sure the latest setvalue applied during 'apply' is reached"""
        # only called if 'apply' has been called beforehand

        if self.sweepmode == "Wavelength in nm":
            # self.wln = self.get_wavelength()
            # if int(self.wln) != int(self.value):
            #     error(f"Wln {self.value} not reached ({self.wln}")
            self._is_mode_locked, tuned = False, False

            # wait loop for tuning
            # print(f"LASER WAITING FOR WAVELENGTH {self.value}")
            refresh_rate, timeout = 0.1, 30
            attempts = int(timeout / refresh_rate) + 1
            start = time.time()
            for _ in range(attempts):
                time.sleep(refresh_rate)
                tuning_status = self.get_tuning_status()
                # ready
                if tuning_status == 0:
                    tuned = True
                    break
                # mode locking or recovery?
                elif tuning_status != 1:
                    # print("laser tuning status: ", tuning_status)
                    pass
            # if tuned:
            #     print(f"Laser tuned in {time.time()-start}s")
            else:
                raise TimeoutError(f"Laser didn't tune within the allocated {timeout}s")

            # second wait loop for modelock
            refresh_rate = 0.3
            attempts = int(self.modelock_timeout / refresh_rate) + 1
            start = time.time()
            for _ in range(attempts):
                time.sleep(refresh_rate)
                modelock_status = self.get_modelock_status()
                if modelock_status == 1:
                    self._is_mode_locked = True
                    break
            # if self._is_mode_locked:
            #     print(f"Laser Modelocked in {time.time()-start}s")
            if not self._is_mode_locked:
                print(f"Warning: Laser didn't modelock within the {self.modelock_timeout}s timeout")

        if self._auto_shutter:
            self.open_shutter()

    def call(self):
        """
        mandatory function that must be used to return as many values as defined in self.variables
        This function can only be omitted if no variables are defined in self.variables.
        """
        return [
            self.get_wavelength(),
            self.get_shutter_state(),
            self.get_power(), self._is_mode_locked
        ]

    def get_power(self) -> float:

        # query = "?UF"
        query = "PRINT UF POWER"
        result = self.query_port(query)
        return float(result)

    def set_shutter_state(self, open_: bool):

        self.write_port(f"SHUTTER={int(open_)}")
        self._is_shutter_open = open_

    def get_shutter_state(self):

        query = "?S"
        # query = "PRINT SHUTTER"
        result = self.query_port(query)
        self._is_shutter_open = bool(result)
        return self._is_shutter_open

    def get_laser_status(self):
        """ check laser status, raises an error if fault found"""
        query = "?L"
        answer = int(self.query_port(query))
        if answer == 2:
            raise ChameleonError("LASER FAULT")
        return answer

    def get_tuning_status(self) -> int:
        """ get the current tuning status of the laser (wln change)
        returns:
            0: ready
            1: tuning, 2: searching modelock, 3: recovery in progress"""
        tuning_status = int(self.query_port("?TS"))
        return tuning_status

    def get_modelock_status(self) -> int:
        """ get the current tuning status of the laser (wln change)
        returns:
            0: Off/standby
            1: Modelocked, 2: CW"""
        modelock_status = int(self.query_port("?MDLK"))
        return modelock_status

    def set_wavelength(self, wln: int):
        """ set the laser wln"""
        self.write_port(f"WAVELENGTH={int(wln):03d}")
        self.wln = wln

    def get_wavelength(self) -> int:
        """get the laser set wln"""
        query = "?VW"
        result = self.query_port(query)
        self.wln = int(result)

        return self.wln

    def set_alignment_mode(self, alignment=True):
        """ enter or exit alignment mode"""
        current_alignment_mode = self.is_in_alignment_mode()
        if current_alignment_mode == alignment:
            print("Desired alignment mode already applied")
            return
        else:
            print(f"applying alignment {'on' if alignment else 'off'}")
        command = f"ALIGN={int(alignment)}"
        self.write_port(command)

    def is_in_alignment_mode(self) -> bool:
        """ enter or exit alignment mode"""
        query = "?ALIGN"
        in_alignment_mode = int(self.query_port(query))

        return bool(in_alignment_mode)

    def home(self):
        """Homes the tuning motor. This action can take 3-30 seconds"""
        self.write_port("HM=1")

    def check_key_turned(self):
        """check laser safety key turned"""
        query = "?K"
        # query = "PRINT KEYSWITCH"
        result = self.query_port(query)
        return bool(result)

    def set_laser_on_standby(self):
        """Turning the keyswitch to STANDBY and then to
        the ON position overrides this command."""
        self.write_port("Laser=0")

    def set_laser_on(self):
        """ Resets faults and turns laser on. Clears fault screen on power supply
            and fault history (?FAULT HISTORY), so lasing resumes if there are no
            active faults."""
        self.write_port("Laser=1")

    def check_errors(self) -> str:
        """ get error list if any and parse it based on manual"""

        query = "?F"
        errors = self.query_port(query)
        if errors == "System OK" or errors == 0:
            return ""

        errors_list = [int(err) for err in errors.split("&")]
        return ",".join([LASER_ERROR_CODES[err] for err in errors_list])

    def close_shutter(self):

        self.set_shutter_state(open_=False)

    def open_shutter(self, check_laser_on=True):

        self.set_shutter_state(open_=True)
        if check_laser_on:
            laser_on = self.get_laser_status()
            if not laser_on:
                raise ChameleonError("Laser is not on")

    def toggle_shutter(self):

        state = self.get_shutter_state()
        self.set_shutter_state(not state)

    def query_port(self, query="") -> str:
        """wrap the query command to deal with the instrument's extra CR/LR after commands"""
        self.port.write(query)
        answer: str = self.port.read()
        # if self.echo_mode and query:
        #     return answer.replace(query + " ", "")
        # else:
        return answer

    def write_port(self, command) -> None:
        """wrap the write command to deal with the instrument's extra CR/LR after commands"""

        self.port.write(command)
        _ = self.port.read()  # read empty string to clear the buffer
        # print(f"Reply to command {command} was '{reply}'")
