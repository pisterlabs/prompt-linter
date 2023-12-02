# -*- coding: utf-8 -*-

# Licensed under the MIT license
# http://opensource.org/licenses/mit-license.php

# Copyright 2009 - Benjamin Kampmann <ben.kampmann@googlemail.com>


from coherence import log
from coherence.upnp.core.utils import parse_xml, getPage, means_true

class MediaRendererClient(log.Loggable):

    def __init__(self, coherence):
        self.coherence = coherence
        self.volume = None
        self.muted = False
        self.device = None

        for name in ['Pause', 'Stop', 'Previous', 'Next']:
            setattr(self, name.lower(), self._proxy(name))

        # play is a bit special
        setattr(self, 'play', self._proxy('Play', Speed=1))

    def connect(self, device):
        assert self.device is None, """Don't connect as long as another device is still connected, stupid"""
        self.device = device
        print "set to", device
        """
        service = self.device.get_service_by_type('RenderingControl')
        service.subscribe_for_variable('Volume',
                callback=lambda var: setattr(self, 'volume', var.value))
        service.subscribe_for_variable('Mute',
                callback=lambda var: setattr(self, 'muted',
                        means_true(var.value)))
        service = self.device.get_service_by_type('AVTransport')
        service.subscribe_for_variable('AVTransportURI',
                callback=lambda var: setattr(self, 'uri', var.value))
        service.subscribe_for_variable('CurrentTrackMetaData',
                callback=lambda var: setattr(self, 'track_metadata', var.value))
        service.subscribe_for_variable('TransportState',
                callback=lambda var: setattr(self, 'state', var.value))
        service.subscribe_for_variable('CurrentTransportActions',
                callback=lambda var: setattr(self, 'actions', var.value))

        service.subscribe_for_variable('AbsTime',
                callback=lambda var: setattr(self, 'abstime', var.value))
        service.subscribe_for_variable('TrackDuration',
                callback=lambda var: setattr(self, 'duration', var.value))
        """


    def _proxy(self, name, **kw):
        def wrapped():
            print name, kw
            service = self.device.get_service_by_type('AVTransport')
            action = service.get_action(name)
            return action.call(InstanceID=0, **kw)
        return wrapped

    def disconnect(self):
        # FIXME: unsubscribe
        pass
