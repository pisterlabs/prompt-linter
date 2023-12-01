import sys, os

from coherence.extern import qt4reactor

from PyQt4 import QtGui, uic
from PyQt4 import QtCore
from PyQt4.QtCore import pyqtRemoveInputHook

from coherence.extern.simple_config import Config
from coherence.extern.telepathy import connect
from telepathy.interfaces import ACCOUNT_MANAGER, ACCOUNT
from telepathy.interfaces import CONN_INTERFACE
from telepathy.constants import CONNECTION_STATUS_CONNECTED, \
     CONNECTION_STATUS_DISCONNECTED, CONNECTION_STATUS_CONNECTING
import dbus

BASEDIR = os.path.dirname(__file__)
UIDIR = os.path.join(BASEDIR, "ui")
CONFIG_PATH = os.path.join(BASEDIR, "mirabeau.xml")

def pdb(locals):
    pyqtRemoveInputHook()
    import pdb; pdb.set_trace()

DEFAULT_CONFIG="""\
<config>
  <serverport>30020</serverport>
  <use_dbus>yes</use_dbus>
  <enable_mirabeau>yes</enable_mirabeau>
  <mirabeau>
   <chatroom>Mirabeau</chatroom>
   <conference-server>conference.jabber.org</conference-server>
   <manager>gabble</manager>
   <protocol>jabber</protocol>
   <account>%(default_account)s</account>
  </mirabeau>
</config>
"""


class UILoader(object):
    uifilename = ""

    def __init__(self, parent=None):
        uifile = os.path.join(UIDIR, self.uifilename)
        self.ui = uic.loadUi(uifile, parent)
        self.connectSignals()

    def __getattr__(self, attr):
        if attr != "ui":
            return getattr(self.ui, attr)

class Window(UILoader):
    uifilename = "main.ui"
    coherence_instance = None

    def __init__(self, CoherenceClass):
        self.CoherenceClass = CoherenceClass
        super(Window, self).__init__()

    def connectSignals(self):
        self.connect(self.settingsButton, QtCore.SIGNAL("clicked()"),
                     self.openSettings)
        # FIXME: reactivate when shutdown/startup of coherence is fixed
        ## self.connect(self.statusButton, QtCore.SIGNAL("clicked()"),
        ##              self.updateStatus)
        self.connect(self.localDevicesButton, QtCore.SIGNAL("clicked()"),
                     self.openLocalDevices)

        self.loadConfig()
        if self.config.get("mirabeau").get("account"):
            self.startCoherence()

    def loadConfig(self):
        if not os.path.exists(CONFIG_PATH):
            try:
                default_account = connect.gabble_accounts()[0]
            except IndexError:
                default_account = ''

            cfg = DEFAULT_CONFIG % locals()
            fd = open(CONFIG_PATH, "w")
            fd.write(cfg)
            fd.close()

        self.config = Config(CONFIG_PATH, root='config')

    def startCoherence(self, restart=False):
        if restart:
            if self.coherence_instance:
                dfr = self.coherence_instance.shutdown()
                dfr.addCallback(lambda result: self.startCoherence())
                return dfr
            else:
                self.coherence_instance = self.CoherenceClass(self.config)
        else:
            self.coherence_instance = self.CoherenceClass(self.config)
        if self.coherence_instance:
            mirabeau_instance = self.coherence_instance.mirabeau
            conn_obj = mirabeau_instance.tube_publisher.conn[CONN_INTERFACE]
            handle = conn_obj.connect_to_signal('StatusChanged',
                                                self.status_changed_cb)
            self.status_update_handle = handle
            #pdb(locals())

    def stopCoherence(self):
        def stopped(result):
            self.coherence_instance.clear()
            self.coherence_instance = None
            print ">>", result
            
        dfr = self.coherence_instance.shutdown()
        dfr.addBoth(stopped)
        return dfr

    def status_changed_cb(self, status, reason):
        if status == CONNECTION_STATUS_CONNECTING:
            text = "Connecting. Please wait"
        elif status == CONNECTION_STATUS_CONNECTED:
            text = 'Connected'
        elif status == CONNECTION_STATUS_DISCONNECTED:
            text = 'Disconnected'
        self.statusButton.setText(text)

    def updateStatus(self):
        if self.coherence_instance:
            self.stopCoherence()
        else:
            self.startCoherence()

    def openSettings(self):
        self._setting_win = Settings(self)
        self._setting_win.show()

    def openLocalDevices(self):
        self._local_devices_win = LocalDevices(self)
        self._local_devices_win.show()

class LocalDevices(UILoader):
    uifilename = "local_devices.ui"

    def __init__(self, parent):
        self.parent = parent
        super(LocalDevices, self).__init__()

    def connectSignals(self):
        listePays = QtCore.QStringList([])
        self.model = QtGui.QStringListModel(listePays)
        self.listView.setModel(self.model)

        coherence = self.parent.coherence_instance
        coherence.connect(self.device_found, 'Coherence.UPnP.RootDevice.detection_completed')
        coherence.connect(self.device_removed, 'Coherence.UPnP.RootDevice.removed')
        for device in coherence.devices:
            self.device_found(device)

    def device_found(self, device=None):
        name = '%s (%s)' % (device.get_friendly_name(), device.get_friendly_device_type())
        devices = self.model.stringList()
        devices.append(name)
        self.model.setStringList(devices)
        #self.model.sort()

    def device_removed(self,usn=None):
        print usn

class Settings(UILoader):
    uifilename = "settings.ui"

    def __init__(self, parent):
        self.parent = parent
        super(Settings, self).__init__()

    def connectSignals(self):
        self.connect(self.buttonBox, QtCore.SIGNAL("accepted()"),
                     self.accepted)
        self.connect(self.buttonBox, QtCore.SIGNAL("rejected()"),
                     self.rejected)

        mirabeau_section = self.parent.config.get("mirabeau")
        self.confServerTextEdit.setText(mirabeau_section.get("conference-server"))
        self.chatRoomTextEdit.setText(mirabeau_section.get("chatroom"))

        # fill accounts box
        bus = dbus.SessionBus()
        model = self.accountsBox.model()
        self.accounts = connect.gabble_accounts()
        for account_obj_path in self.accounts:
            account_obj = bus.get_object(ACCOUNT_MANAGER, account_obj_path)
            norm_name = account_obj.Get(ACCOUNT, 'NormalizedName')
            nick_name = account_obj.Get(ACCOUNT, 'Nickname')
            label = "%s - %s" % (nick_name, norm_name)
            model.appendRow(QtGui.QStandardItem(label))

        #pdb(locals())

        conf_account = mirabeau_section.get("account")
        if conf_account and conf_account in self.accounts:
            index = self.accounts.index(conf_account)
            self.accountsBox.setCurrentIndex(index)


    def accepted(self):
        mirabeau_section = self.parent.config.get("mirabeau")
        mirabeau_section.set("chatroom",
                             str(self.chatRoomTextEdit.text()))
        mirabeau_section.set("conference-server",
                             str(self.confServerTextEdit.text()))
        mirabeau_section.set("account",
                             self.accounts[self.accountsBox.currentIndex()])
        self.parent.config.set("mirabeau", mirabeau_section)
        self.parent.config.save()
        self.parent.loadConfig()

        # TODO: send notification using dbus

    def rejected(self):
        # FIXME: remove this if not used later on.
        pass

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    qt4reactor.install()

    from twisted.internet import reactor
    from coherence.base import Coherence

    def start():
        print "start"
        win = Window(Coherence)
        win.show()
        app.exec_()
        reactor.stop()

    reactor.callWhenRunning(start)
    reactor.run()
