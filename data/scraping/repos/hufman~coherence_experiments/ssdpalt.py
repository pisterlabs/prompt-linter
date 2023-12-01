from coherence import log
from coherence.upnp.core.ssdp import SSDPServer, SSDP_ADDR, SSDP_PORT

from twisted.internet import error, reactor, task

class SSDPServerAlt(SSDPServer):
    """ A subclass of the Coherence SSDPServer object
    It sends broadcasts from a random port, instead of port 1900, per spec. """

    def __init__(self, test=False, interface=''):
        # Create SSDP server
        log.Loggable.__init__(self)
        self.known = {}
        self._callbacks = {}
        self.test = test
        if self.test == False:
            try:
                self.port = reactor.listenMulticast(0, self, listenMultiple=True)
                #self.port.setLoopbackMode(1)

                self.port.joinGroup(SSDP_ADDR, interface=interface)

                self.resend_notify_loop = task.LoopingCall(self.resendNotify)
                self.resend_notify_loop.start(777.0, now=False)

                self.check_valid_loop = task.LoopingCall(self.check_valid)
                self.check_valid_loop.start(333.0, now=False)

            except error.CannotListenError, err:
                self.error("Error starting the SSDP-server: %s", err)
                self.error("There seems to be already a SSDP server running on this host, no need starting a second one.")


