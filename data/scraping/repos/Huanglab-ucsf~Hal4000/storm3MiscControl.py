#!/usr/bin/python
#
# Miscellaneous controls, such as the EPI/TIRF motor and
# the various lasers for STORM3.
#
# Hazen 12/09
#

import sys
from PyQt4 import QtCore, QtGui

# Debugging
import halLib.hdebug as hdebug

# UIs.
import qtdesigner.misccontrolsui_v1 as miscControlsUi

# SMC100 motor (for EPI/TIRF)
import newport.SMC100 as SMC100

# Compass315M 532nm laser control
import coherent.compass315M as compass315M

# Innova 70C control
import coherent.innova70C as innova70C


#
# Thread for communicating with the Innova laser,
# which is really slow.
#
class QInnovaThread(QtCore.QThread):
    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
        self.innova = innova70C.Innova70C()
        self.mutex = QtCore.QMutex()
        self.running = 1

    def getCurrent(self):
        if self.innova.live:
            self.mutex.lock()
            current = self.innova.getCurrent()
            self.mutex.unlock()
            return current
        else:
            return 0.0

    def getLaserOnOff(self):
        if self.innova.live:
            self.mutex.lock()
            on_off = self.innova.getLaserOnOff()
            self.mutex.unlock()
            return on_off
        else:
            return 0

    def innovaQuit(self):
        self.quit()
        if self.innova.live:
            self.innova.shutDown()

    def run(self):
        self.count = 0
        while self.running and self.innova.live:
            #
            # If thread stays asleep for long periods of time (i.e. 500ms) then
            # it can appear as though the program is hung when exiting. To avoid
            # this problem, the thread wakes up every 50ms to see whether it needs
            # to exit, but only actually talks to the laser every 10th time it wakes.
            #
            if self.count == 10:
                self.mutex.lock()
                light = self.innova.getLight()
                hours = self.innova.getHours()
                temp = self.innova.getTemperature()
                self.mutex.unlock()
                self.emit(QtCore.SIGNAL("innovaUpdate(float, float, float)"), light, hours, temp)
                self.count = 0
            self.count += 1
            self.msleep(50)

    def setPower(self, power):
        if self.innova.live:
            self.mutex.lock()
            self.innova.setLaserCurrent(power)
            self.mutex.unlock()

    def stopThread(self):
        if self.innova.live:
            self.running = 0
            



#
# Misc Control Dialog Box
#
class AMiscControl(QtGui.QDialog):
    @hdebug.debug
    def __init__(self, parameters, parent = None):
        QtGui.QMainWindow.__init__(self, parent)
        self.debug = 1
        self.move_timer = QtCore.QTimer(self)
        self.move_timer.setInterval(50)
        if parent:
            self.have_parent = 1
        else:
            self.have_parent = 0

        # UI setup
        self.ui = miscControlsUi.Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle(parameters.setup_name + " Misc Control")

        self.ui.dYAGButton.hide()
        self.ui.arKrButton.hide()

        # connect signals
        if self.have_parent:
            self.ui.okButton.setText("Close")
            self.connect(self.ui.okButton, QtCore.SIGNAL("clicked()"), self.handleOk)
        else:
            self.ui.okButton.setText("Quit")
            self.connect(self.ui.okButton, QtCore.SIGNAL("clicked()"), self.handleQuit)

        self.connect(self.ui.EPIButton, QtCore.SIGNAL("clicked()"), self.goToEPI)
        self.connect(self.ui.leftSmallButton, QtCore.SIGNAL("clicked()"), self.smallLeft)
        self.connect(self.ui.rightSmallButton, QtCore.SIGNAL("clicked()"), self.smallRight)
        self.connect(self.ui.leftLargeButton, QtCore.SIGNAL("clicked()"), self.largeLeft)
        self.connect(self.ui.rightLargeButton, QtCore.SIGNAL("clicked()"), self.largeRight)
        self.connect(self.ui.TIRFButton, QtCore.SIGNAL("clicked()"), self.goToTIRF)
        self.connect(self.ui.tirGoButton, QtCore.SIGNAL("clicked()"), self.goToX)
        self.connect(self.ui.dYAGSlider, QtCore.SIGNAL("valueChanged(int)"), self.dYAGChange)
        self.connect(self.move_timer, QtCore.SIGNAL("timeout()"), self.updatePosition)
        self.connect(self.ui.dYAGButton, QtCore.SIGNAL("clicked()"), self.changedYAGPower)
        self.connect(self.ui.arKrButton, QtCore.SIGNAL("clicked()"), self.changeArKrPower)

        # set modeless
        self.setModal(False)

        self.jog_size = 0.05
        self.epi_position = 18.6
        self.tirf_position = 21.3
        self.dYAG_max = 165.0
        self.dYAG_power = 40.0
        self.old_dYAG_power = -1.0
        if parameters:
            self.newParameters(parameters)

        # epi/tir stage init
        self.smc100 = SMC100.SMC100()
        self.position = self.smc100.getPosition()
#        if self.position < 18.0:
#            self.position = self.epi_position
#            self.move()
#        if self.position > 22.0:
#            self.position = self.tirf_position
#            self.move()
        self.setPositionText()

        # compass laser init
        self.compass = compass315M.Compass315M(board = "PCI-MIO-16E-4")
        self.compass.start(self.dYAG_power/self.dYAG_max)
        self.ui.dYAGSlider.setSliderPosition(int((self.dYAG_power/self.dYAG_max)*float(self.ui.dYAGSlider.maximum())))
        self.changedYAGPower()

        # innova70 init
        self.innova = QInnovaThread()
        if not self.innova.getLaserOnOff():
            print "The Innova Laser appears to be turned off."
        self.arkrpower = self.innova.getCurrent()
        self.ui.arKrAmps.setText("{0:.1f} Amps".format(self.arkrpower))
        self.ui.arKrSlider.setValue(int(self.arkrpower))
        self.connect(self.innova, QtCore.SIGNAL("innovaUpdate(float, float, float)"), self.updateArKrStats)
        self.innova.start(QtCore.QThread.NormalPriority)

        # we connect this last so that setting the current power doesn't
        # throw up the update button.
        self.connect(self.ui.arKrSlider, QtCore.SIGNAL("valueChanged(int)"), self.arKrChange)

    @hdebug.debug
    def arKrChange(self, value):
        self.arkrpower = value
        self.ui.arKrAmps.setText("{0:.1f} Amps".format(self.arkrpower))
        self.ui.arKrButton.show()

    @hdebug.debug
    def changeArKrPower(self):
        self.innova.setLaserCurrent(self.arkrpower)
        self.ui.arKrButton.hide()

    @hdebug.debug
    def changedYAGPower(self):
        if (self.old_dYAG_power != self.dYAG_power):
            self.compass.setPower(self.dYAG_power/self.dYAG_max)
            self.old_dYAG_power = self.dYAG_power
        self.ui.dYAGButton.hide()

    @hdebug.debug
    def closeEvent(self, event):
        if self.have_parent:
            event.ignore()
            self.hide()
        else:
            self.quit()

    @hdebug.debug
    def dYAGChange(self, value):
        self.dYAG_power = (float(value)/float(self.ui.dYAGSlider.maximum())) * self.dYAG_max
        self.ui.dYAGText.setText(str(self.dYAG_power) + " mw")
        self.ui.dYAGButton.show()

    @hdebug.debug
    def goToEPI(self):
        self.position = self.epi_position
        self.moveStage()

    @hdebug.debug
    def goToTIRF(self):
        self.position = self.tirf_position
        self.moveStage()

    @hdebug.debug
    def goToX(self):
        self.position = self.ui.tirSpinBox.value()
        self.moveStage()

    @hdebug.debug
    def handleOk(self):
        self.hide()

    @hdebug.debug
    def handleQuit(self):
        self.close()

    @hdebug.debug
    def largeLeft(self):
        if self.position > 14.0:
            self.position -= 10.0 * self.jog_size
            self.moveStage()

    @hdebug.debug
    def largeRight(self):
        if self.position < 23.0:
            self.position += 10.0 * self.jog_size
            self.moveStage()

    def moveStage(self):
        self.move_timer.start()
        self.smc100.stopMove()
        self.smc100.moveTo(self.position)
        self.setPositionText()

    @hdebug.debug
    def newParameters(self, parameters):
        self.debug = parameters.debug
        self.jog_size = parameters.jog_size
        self.epi_position = parameters.epi_position
        self.tirf_position = parameters.tirf_position
        self.dYAG_max = parameters.dYAG_max
        self.dYAG_power = parameters.dYAG_power

    def setPositionText(self):
        self.ui.positionText.setText("{0:.3f}".format(self.position))

    @hdebug.debug
    def smallLeft(self):
        if self.position > 14.0:
            self.position -= self.jog_size
            self.moveStage()

    @hdebug.debug
    def smallRight(self):
        if self.position < 23.0:
            self.position += self.jog_size
            self.moveStage()

    def updateArKrStats(self, light, hours, temp):
        self.ui.arKrWatts.setText("{0:.2f} W".format(light))
        self.ui.arKrHours.setText("{0:.2f} Hrs".format(hours))
        self.ui.arKrTemp.setText("{0:.1f} C".format(temp))

    def updatePosition(self):
        if not self.smc100.amMoving():
            self.move_timer.stop()
        self.position = self.smc100.getPosition()
        self.setPositionText()

    @hdebug.debug
    def quit(self):
        if hdebug.getDebug():
            print "  compass.stop"
        self.compass.stop()
        if hdebug.getDebug():
            print "  stopThread"
        self.innova.stopThread()
        if hdebug.getDebug():
            print "  wait"
        self.innova.wait(200)
        if hdebug.getDebug():
            print "  quit"
        self.innova.innovaQuit()


#
# testing
#

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    miscControl = MiscControl()
    miscControl.show()
    app.exec_()


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
