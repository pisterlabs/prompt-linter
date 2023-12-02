#!/usr/bin/python
#
# Illumination control specialized for Sequoia (RJM 4/5/2014)
#
# Hazen 6/09
#

from PyQt4 import QtCore

import illumination.channelWidgets as channelWidgets
import illumination.commandQueues as commandQueues
import illumination.illuminationControl as illuminationControl
import illumination.shutterControl as shutterControl
import nationalInstruments.nicontrol as nicontrol

import coherent.obis as obis

#
# Illumination power control specialized for Acadia (copied from Yosemite).
#
class SequoiaQIlluminationControlWidget(illuminationControl.QIlluminationControlWidget):
    def __init__(self, settings_file_name, parameters, parent = None):
        print "init Sequoia IC widget"
        # setup the AOTF communication thread
        #self.aotf_queue = commandQueues.QAOTFThread()
        #self.aotf_queue.start(QtCore.QThread.NormalPriority)

        # setup the Obis642 communication thread
        self.obis642_queue = commandQueues.QSerialLaserComm(obis.Obis(port="COM5"))
        self.obis642_queue.start(QtCore.QThread.NormalPriority)

        # setup the Obis405 communication thread
        self.obis405_queue = commandQueues.QSerialLaserComm(obis.Obis(port="COM3"))
        self.obis405_queue.start(QtCore.QThread.NormalPriority)

        # setup the Obis488 communication thread
        self.obis488_queue = commandQueues.QSerialLaserComm(obis.Obis(port="COM4"))
        self.obis488_queue.start(QtCore.QThread.NormalPriority)

        # set the ThorlabsLED communication Thread
        self.thorlabsLED_queue = commandQueues.QThorlabsLEDThread()   
        self.thorlabsLED_queue.start(QtCore.QThread.NormalPriority)

        # set the Sapphire-AOM frequency
        self.sapphire561_queue = commandQueues.QSapphTESTThread()
        self.sapphire561_queue.start(QtCore.QThread.NormalPriority)

        # setup the AOM communication thread
        #self.aom_queue = commandQueues.QNiAnalogComm(5.0)  # 2/26/10 added
        #self.aom_queue.start(QtCore.QThread.NormalPriority)

        # setup for NI communication (analog, camera synced)
        self.ni_queue = commandQueues.QNiAnalogComm(5.0)

        # setup for NI communication with mechanical backup shutters (digital, unsynced)
        self.shutter_queue = commandQueues.QNiDigitalComm()

        illuminationControl.QIlluminationControlWidget.__init__(self, settings_file_name, parameters, parent)

    def autoControl(self, channels):
        print "autoControl"
        #self.aotf_queue.analogModulationOn()
        #self.cube_queue.analogModulationOff()
        self.ni_queue.setFilming(1)
        self.shutter_queue.setFilming(1)
        illuminationControl.QIlluminationControlWidget.autoControl(self, channels)

    def manualControl(self):
        print "manualControl"
        #self.aotf_queue.analogModulationOff()
        self.ni_queue.setFilming(0)
        self.shutter_queue.setFilming(0)
        illuminationControl.QIlluminationControlWidget.manualControl(self)

    def newParameters(self, parameters):
        illuminationControl.QIlluminationControlWidget.newParameters(self, parameters)

        # Set the size based on the number of channels
        dx = 50
        width = self.number_channels * dx

        # The height is based on how many buttons there are per channel,
        # so first we figure out the number of buttons per channel.
        max_buttons = 0
        for i in range(self.number_channels):
            n_buttons = len(parameters.power_buttons[i])
            if n_buttons > max_buttons:
                max_buttons = n_buttons
        height = 204 + 18 + max_buttons * 22

        self.resize(width, height)
        self.setMinimumSize(QtCore.QSize(width, height))
        self.setMaximumSize(QtCore.QSize(width, height))

        # Create the individual channels
        x = 0
        for i in range(self.number_channels):
            n = self.settings[i].channel
            if hasattr(self.settings[i], 'use_aotf'):
                channel = channelWidgets.QAOTFChannel(self,
                                                              self.settings[i],
                                                              parameters.default_power[n],
                                                              parameters.on_off_state[n],
                                                              parameters.power_buttons[n],
                                                              x,
                                                              dx,
                                                              height)
                channel.setCmdQueue(self.aotf_queue)
                channel.setShutterQueue(self.shutter_queue)
                self.channels.append(channel)
            elif hasattr(self.settings[i], 'use_obis405'):               # 2/26/10 added
                channel = channelWidgets.QObisChannelWShutter(self,
                                                                 self.settings[i],
                                                                 parameters.default_power[n],
                                                                 parameters.on_off_state[n],
                                                                 parameters.power_buttons[n],
                                                                 x,
                                                                 dx,
                                                                 height)
                channel.setCmdQueue(self.obis405_queue)
                channel.setShutterQueue(self.shutter_queue)
                self.channels.append(channel)
            elif hasattr(self.settings[i], 'use_obis488'):               # 2/26/10 added
                channel = channelWidgets.QObisChannelWShutter(self,
                                                                 self.settings[i],
                                                                 parameters.default_power[n],
                                                                 parameters.on_off_state[n],
                                                                 parameters.power_buttons[n],
                                                                 x,
                                                                 dx,
                                                                 height)
                channel.setCmdQueue(self.obis488_queue)
                channel.setShutterQueue(self.shutter_queue)
                self.channels.append(channel)
            elif hasattr(self.settings[i], 'use_obis642'):               # 2/26/10 added
                channel = channelWidgets.QObisChannelWShutter(self,
                                                                 self.settings[i],
                                                                 parameters.default_power[n],
                                                                 parameters.on_off_state[n],
                                                                 parameters.power_buttons[n],
                                                                 x,
                                                                 dx,
                                                                 height)
                channel.setCmdQueue(self.obis642_queue)
                channel.setShutterQueue(self.shutter_queue)
                self.channels.append(channel)
            elif hasattr(self.settings[i], 'use_sapphire561'):               # 4/2/2014 added
                print "Using Sapphire561 on channel: ", n
                channel = channelWidgets.QSapphTESTChannelWShutter(self,
                                                                 self.settings[i],
                                                                 parameters.default_power[n],
                                                                 parameters.on_off_state[n],
                                                                 parameters.power_buttons[n],
                                                                 x,
                                                                 dx,
                                                                 height)
                channel.setCmdQueue(self.sapphire561_queue)
                channel.setShutterQueue(self.shutter_queue)
                self.channels.append(channel)
            elif hasattr(self.settings[i], 'use_thorlabsLED'):              # 8/02/10 added
                channel = channelWidgets.QThorlabsLEDChannelWShutter(self,
                                                                 self.settings[i],
                                                                 parameters.default_power[n],
                                                                 parameters.on_off_state[n],
                                                                 parameters.power_buttons[n],
                                                                 x,
                                                                 dx,
                                                                 height)
                channel.setCmdQueue(self.thorlabsLED_queue)
                channel.setShutterQueue(self.shutter_queue)
                self.channels.append(channel)
            elif hasattr(self.settings[i], 'mechanical_shutter'):
                channel = channelWidgets.QNIChannel(self,
                                                    self.settings[i],
                                                    parameters.on_off_state[n],
                                                    x,
                                                    dx,
                                                    height)
                channel.setCmdQueue(self.ni_queue)
                self.channels.append(channel)
            x += dx

        # Update the channels to reflect their current ui settings.
        for channel in self.channels:
            channel.uiUpdate()

        # Save access to the previous parameters file so that
        # we can save the settings when the parameters are changed.
        self.last_parameters = parameters

    def shutDown(self):
        print "Acadia shutdown"
        illuminationControl.QIlluminationControlWidget.shutDown(self)
        #self.aotf_queue.stopThread()
        #self.aotf_queue.wait()
        self.obis642_queue.stopThread()
        self.obis642_queue.wait()
        self.obis405_queue.stopThread()
        self.obis405_queue.wait()
        self.obis488_queue.stopThread()
        self.obis488_queue.wait()
        self.sapphire561_queue.stopThread()
        self.sapphire561_queue.wait()





class SequoiaShutterControl(shutterControl.ShutterControl):
    def __init__(self, powerToVoltage, parent):
        shutterControl.ShutterControl.__init__(self, powerToVoltage, parent)
        self.ct_task = 0
        self.wv_task = 0
        self.board = "PCIe-6353"
        self.oversampling = 100
        self.number_channels = 5                                       # channel 6 is LED, added 8/02/10


    def cleanup(self):
        if self.ct_task:
            self.ct_task.clearTask()
            self.wv_task.clearTask()
            self.ct_task = 0
            self.wv_task = 0

    def setup(self):
        assert self.ct_task == 0, "Attempt to call setup without first calling cleanup."
        #
        # the counter runs slightly faster than the camera so that it is ready
        # to catch the next camera "fire" immediately after the end of the cycle.
        #
        frequency = (1.001/self.kinetic_value) * float(self.oversampling)
        print "self.kinetic_value: ", self.kinetic_value
        print "self.oversampling: ", self.oversampling
        print "Setup shutter sequence with frequency: ", frequency

        # set up the analog channels
        #self.wv_task = nicontrol.WaveformOutput(self.board, 0)
        #for i in range(self.number_channels - 1):
        #    self.wv_task.addChannel(i + 1)

        # set up the digital channels
        self.wv_task = nicontrol.DigitalWaveformOutput(self.board, 0) #Was DigWaveformOutput (changed 4/16/2014)
        for i in range(self.number_channels - 1):
            self.wv_task.addChannel(self.board,i + 1) #was addDigChannel

        # set up the waveform
        self.wv_task.setWaveform(self.waveforms, frequency) #Was setDigWaveform (changed 4/16/2014)

        # set up the counter
        self.ct_task = nicontrol.CounterOutput(self.board, 0, frequency, 0.5)
        self.ct_task.setCounter(self.waveform_len)
        self.ct_task.setTrigger(0)

    def startFilm(self):
        self.wv_task.startTask()
        self.ct_task.startTask()

    def stopFilm(self):
        # stop the tasks
        if self.ct_task:
            self.ct_task.stopTask()
            self.wv_task.stopTask()
            self.ct_task.clearTask()
            self.wv_task.clearTask()
            self.ct_task = 0
            self.wv_task = 0

        # reset all the analog signals.
        #for i in range(self.number_channels):
        #    ao_task = nicontrol.VoltageOutput(self.board, i)
        #    ao_task.outputVoltage(self.powerToVoltage(i, 0.0))
        #    ao_task.startTask()
        #    ao_task.stopTask()
        #    ao_task.clearTask()

        # reset all the digital signals.
        for i in range(self.number_channels): #we have exactly 1 channel at all times
            do_task = nicontrol.DigitalOutput(self.board, i)
            do_task.output(0)
            do_task.startTask()
            do_task.stopTask()
            do_task.clearTask()

#
# Illumination power control dialog box specialized for Sequoia).
#
class AIlluminationControl(illuminationControl.IlluminationControl):
    #def __init__(self, parameters, tcp_control, parent = None):
    def __init__(self, hardware, parameters, parent=None):
        illuminationControl.IlluminationControl.__init__(self, parameters, parent) #No tcp
        self.power_control = SequoiaQIlluminationControlWidget("illumination/sequoia_illumination_control_settings.xml",
                                                              parameters,
                                                              parent = self.ui.laserBox)
        self.shutter_control = SequoiaShutterControl(self.power_control.powerToVoltage,
                                                     self.ui.laserBox)
        self.updateSize()

#    def turnOnOff(self, channels, on):
#        if self.debug:
#            print " turnOnOff"
#        if on:
#            self.power_control.allOff()
#        else:
#            self.power_control.turnOnOff(channels, on)

#
# The MIT License
#
# Copyright (c) 2009 Zhuang Lab, Harvard University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
