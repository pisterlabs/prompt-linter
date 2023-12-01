# Licensed under the MIT license
# http://opensource.org/licenses/mit-license.php

# Copyright 2009 Frank Scholz <coherence@beebits.net>

import os
import platform
import sys

try:
    _ = set()
except:
    from sets import Set as set

# Twisted
from twisted.internet import reactor
from twisted.internet import defer
from twisted.web import client
from twisted.python import failure

# Coherence
from coherence.upnp.core.soap_service import errorCode

from coherence.upnp.core import DIDLLite

import coherence.extern.louie as louie
from coherence import log


class CadreRenderer(log.Loggable):

    logCategory = 'renderer'

    implements = ['MediaRenderer']

    vendor_value_defaults = {'AVTransport': {'CurrentPlayMode': ('NORMAL', 'REPEAT_ALL', )}}

    def __init__(self, device, **kwargs):
        #print "PictureRenderer", kwargs
        self.server = device
        try:
            self.name = kwargs['name']
        except KeyError:
            self.name = "PictureFrame on %s" % self.server.coherence.hostname

        self.controller = kwargs['controller']

        try:
            self.display_time = kwargs['display_time']
        except KeyError:
            self.display_time = 20

        try:
            self.display_transition = kwargs['transition']
        except KeyError:
            self.display_transition = 'NONE'

        self.playcontainer = None

        self.auto_next_image = None
        self.display_loop = None

        self.dlna_caps = ['playcontainer-0-1']

        louie.send('Coherence.UPnP.Backend.init_completed', None, backend=self)

    def __repr__(self):
        return str(self.__class__).split('.')[-1]

    def got_image(self, result, title=''):
        """ we have a new image, pass it to the texture """
        self.controller.canvas.show_image(result, title)

    def start_auto_next_image(self):
        self.stop_auto_next_image()
        connection_id = self.server.connection_manager_server.lookup_avt_id(self.current_connection_id)
        if self.server.av_transport_server.get_variable('CurrentPlayMode').value in ['REPEAT_ALL']:
            self.display_loop = reactor.callLater(self.display_time, self.upnp_Next, InstanceID=0)

    def stop_auto_next_image(self):
        if self.auto_next_image:
            self.auto_next_image.cancel()
        self.auto_next_image = None

    def load(self, uri, metadata, mimetype=None):
        was_playing = self.playing
        #if was_playing == True:
        #    self.stop()

        self.playing = False
        self.metadata = metadata

        connection_id = self.server.connection_manager_server.lookup_avt_id(self.current_connection_id)

        if self.playcontainer == None:
            self.server.av_transport_server.set_variable(connection_id, 'AVTransportURI', uri)
            self.server.av_transport_server.set_variable(connection_id, 'AVTransportURIMetaData', metadata)
            self.server.av_transport_server.set_variable(connection_id, 'NumberOfTracks', 1)
            self.server.av_transport_server.set_variable(connection_id, 'CurrentTrack', 1)
        else:
            self.server.av_transport_server.set_variable(connection_id, 'AVTransportURI', self.playcontainer[1])
            self.server.av_transport_server.set_variable(connection_id, 'NumberOfTracks', len(self.playcontainer[2]))
            self.server.av_transport_server.set_variable(connection_id, 'CurrentTrack', self.playcontainer[0] + 1)

        self.server.av_transport_server.set_variable(connection_id, 'CurrentTrackURI', uri)
        self.server.av_transport_server.set_variable(connection_id, 'CurrentTrackMetaData', metadata)

        transport_actions = set(['PLAY,STOP,PAUSE'])
        if len(self.server.av_transport_server.get_variable('NextAVTransportURI').value) > 0:
            transport_actions.add('NEXT')

        if self.playcontainer != None:
            if len(self.playcontainer[2]) - (self.playcontainer[0] + 1) > 0:
                transport_actions.add('NEXT')
            if self.playcontainer[0] > 0:
                transport_actions.add('PREVIOUS')

        self.server.av_transport_server.set_variable(connection_id, 'CurrentTransportActions', transport_actions)

        if was_playing == True:
            self.play()

    def stop(self):
        self.stop_auto_next_image()
        #if self.displaying is not None:
        #    self.controller.display_clear_part(self.displaying)
        self.playing = False
        self.server.av_transport_server.set_variable( \
            self.server.connection_manager_server.lookup_avt_id(self.current_connection_id), \
                             'TransportState', 'STOPPED')

    def play(self):
        def got_error(error, url):
            print "got_error", error, url

        self.playing = True
        connection_id = self.server.connection_manager_server.lookup_avt_id(self.current_connection_id)
        image_url = self.server.av_transport_server.get_variable('CurrentTrackURI').value
        metadata = self.server.av_transport_server.get_variable('CurrentTrackMetaData').value
        image_title = ''
        if metadata != None and len(metadata) > 0:
            elt = DIDLLite.DIDLElement.fromString(metadata)
            item = elt.getItems()[0]
            image_title = item.title
        self.stop_auto_next_image()
        if image_url.startswith("file://"):
            self.got_image(image_url)
            self.start_auto_next_image()
        else:
            d = client.getPage(image_url, timeout=111)
            d.addCallback(self.got_image, image_title)
            d.addCallback(lambda x: self.start_auto_next_image())
            d.addErrback(got_error, image_url)
            d.addErrback(got_error, image_url)

        self.server.av_transport_server.set_variable(connection_id, 'TransportState', 'PLAYING')

    def playcontainer_browse(self, uri):
        """
        dlna-playcontainer://uuid%3Afe814e3e-5214-4c24-847b-383fb599ff01?sid=urn%3Aupnp-org%3AserviceId%3AContentDirectory&cid=1441&fid=1444&fii=0&sc=&md=0
        """
        from urllib import unquote
        from cgi import parse_qs
        from coherence.extern.et import ET
        from coherence.upnp.core.utils import parse_xml

        def handle_reply(r, uri, action, kw):
            try:
                next_track = ()
                elt = DIDLLite.DIDLElement.fromString(r['Result'])
                item = elt.getItems()[0]
                local_protocol_infos = self.server.connection_manager_server.get_variable('SinkProtocolInfo').value.split(',')
                res = item.res.get_matching(local_protocol_infos, protocol_type='internal')
                if len(res) == 0:
                    res = item.res.get_matching(local_protocol_infos)
                if len(res) > 0:
                    res = res[0]
                    remote_protocol, remote_network, remote_content_format, _ = res.protocolInfo.split(':')
                    didl = DIDLLite.DIDLElement()
                    didl.addItem(item)
                    next_track = (res.data, didl.toString(), remote_content_format)
                """ a list with these elements:

                    the current track index
                     - will change during playback of the container items
                    the initial complete playcontainer-uri
                    a list of all the items in the playcontainer
                    the action methods to do the Browse call on the device
                    the kwargs for the Browse call
                     - kwargs['StartingIndex'] will be modified during further Browse requests
                """
                self.playcontainer = [int(kw['StartingIndex']), uri, elt.getItems()[:], action, kw]

                def browse_more(starting_index, number_returned, total_matches):
                    self.info("browse_more %s %s %s", starting_index, number_returned, total_matches)
                    try:

                        def handle_error(r):
                            pass

                        def handle_reply(r, starting_index):
                            elt = DIDLLite.DIDLElement.fromString(r['Result'])
                            self.playcontainer[2] += elt.getItems()[:]
                            browse_more(starting_index, int(r['NumberReturned']), int(r['TotalMatches']))

                        if((number_returned != 5 or
                           number_returned < (total_matches - starting_index)) and
                            (total_matches - number_returned) != starting_index):
                            self.info("seems we have been returned only a part of the result")
                            self.info("requested %d, starting at %d", 5, starting_index)
                            self.info("got %d out of %d", number_returned, total_matches)
                            self.info("requesting more starting now at %d", starting_index + number_returned)
                            self.playcontainer[4]['StartingIndex'] = str(starting_index + number_returned)
                            d = self.playcontainer[3].call(**self.playcontainer[4])
                            d.addCallback(handle_reply, starting_index + number_returned)
                            d.addErrback(handle_error)
                    except:
                        import traceback
                        traceback.print_exc()

                browse_more(int(kw['StartingIndex']), int(r['NumberReturned']), int(r['TotalMatches']))

                if len(next_track) == 3:
                    return next_track
            except:
                import traceback
                traceback.print_exc()

            return failure.Failure(errorCode(714))

        def handle_error(r):
            return failure.Failure(errorCode(714))

        try:
            udn, args = uri[21:].split('?')
            udn = unquote(udn)
            args = parse_qs(args)

            type = args['sid'][0].split(':')[-1]

            try:
                sc = args['sc'][0]
            except:
                sc = ''

            device = self.server.coherence.get_device_with_id(udn)
            service = device.get_service_by_type(type)
            action = service.get_action('Browse')

            kw = {'ObjectID': args['cid'][0],
                  'BrowseFlag': 'BrowseDirectChildren',
                  'StartingIndex': args['fii'][0],
                  'RequestedCount': str(5),
                  'Filter': '*',
                  'SortCriteria': sc}

            d = action.call(**kw)
            d.addCallback(handle_reply, uri, action, kw)
            d.addErrback(handle_error)
            return d
        except:
            return failure.Failure(errorCode(714))

    def upnp_Play(self, *args, **kwargs):
        InstanceID = int(kwargs['InstanceID'])
        Speed = int(kwargs['Speed'])
        self.play()
        return {}

    def upnp_Stop(self, *args, **kwargs):
        InstanceID = int(kwargs['InstanceID'])
        self.stop()
        return {}

    def upnp_Next(self, *args, **kwargs):
        InstanceID = int(kwargs['InstanceID'])
        track_nr = self.server.av_transport_server.get_variable('CurrentTrack')
        return self.upnp_Seek(self, InstanceID=InstanceID, Unit='TRACK_NR', Target=str(int(track_nr.value) + 1))

    def upnp_Previous(self, *args, **kwargs):
        InstanceID = int(kwargs['InstanceID'])
        track_nr = self.server.av_transport_server.get_variable('CurrentTrack')
        return self.upnp_Seek(self, InstanceID=InstanceID, Unit='TRACK_NR', Target=str(int(track_nr.value) - 1))

    def upnp_Seek(self, *args, **kwargs):
        InstanceID = int(kwargs['InstanceID'])
        Unit = kwargs['Unit']
        Target = kwargs['Target']
        if InstanceID != 0:
            return failure.Failure(errorCode(718))
        if Unit in ['TRACK_NR']:
            if self.playcontainer == None:
                NextURI = self.server.av_transport_server.get_variable('NextAVTransportURI').value
                if NextURI != '':
                    self.server.av_transport_server.set_variable(InstanceID, 'TransportState', 'TRANSITIONING')
                    NextURIMetaData = self.server.av_transport_server.get_variable('NextAVTransportURIMetaData').value
                    self.server.av_transport_server.set_variable(InstanceID, 'NextAVTransportURI', '')
                    self.server.av_transport_server.set_variable(InstanceID, 'NextAVTransportURIMetaData', '')
                    r = self.upnp_SetAVTransportURI(self, InstanceID=InstanceID, CurrentURI=NextURI, CurrentURIMetaData=NextURIMetaData)
                    return r
            else:
                Target = int(Target)
                if 0 < Target <= len(self.playcontainer[2]):
                    self.server.av_transport_server.set_variable(InstanceID, 'TransportState', 'TRANSITIONING')
                    next_track = ()
                    item = self.playcontainer[2][Target - 1]
                    local_protocol_infos = self.server.connection_manager_server.get_variable('SinkProtocolInfo').value.split(',')
                    res = item.res.get_matching(local_protocol_infos, protocol_type='internal')
                    if len(res) == 0:
                        res = item.res.get_matching(local_protocol_infos)
                    if len(res) > 0:
                        res = res[0]
                        remote_protocol, remote_network, remote_content_format, _ = res.protocolInfo.split(':')
                        didl = DIDLLite.DIDLElement()
                        didl.addItem(item)
                        next_track = (res.data, didl.toString(), remote_content_format)
                        self.playcontainer[0] = Target - 1

                    if len(next_track) == 3:
                        self.server.av_transport_server.set_variable(self.server.connection_manager_server.lookup_avt_id(self.current_connection_id), 'CurrentTrack', Target)
                        self.load(next_track[0], next_track[1], next_track[2])
                        self.play()
                        return {}
            return failure.Failure(errorCode(711))

        return failure.Failure(errorCode(710))

    def upnp_SetNextAVTransportURI(self, *args, **kwargs):
        InstanceID = int(kwargs['InstanceID'])
        NextURI = kwargs['NextURI']
        current_connection_id = self.server.connection_manager_server.lookup_avt_id(self.current_connection_id)
        NextMetaData = kwargs['NextURIMetaData']
        self.server.av_transport_server.set_variable(current_connection_id, 'NextAVTransportURI', NextURI)
        self.server.av_transport_server.set_variable(current_connection_id, 'NextAVTransportURIMetaData', NextMetaData)
        if len(NextURI) == 0 and self.playcontainer == None:
            transport_actions = self.server.av_transport_server.get_variable('CurrentTransportActions').value
            transport_actions = set(transport_actions.split(','))
            try:
                transport_actions.remove('NEXT')
                self.server.av_transport_server.set_variable(current_connection_id, 'CurrentTransportActions', transport_actions)
            except KeyError:
                pass
            return {}
        transport_actions = self.server.av_transport_server.get_variable('CurrentTransportActions').value
        transport_actions = set(transport_actions.split(','))
        transport_actions.add('NEXT')
        self.server.av_transport_server.set_variable(current_connection_id, 'CurrentTransportActions', transport_actions)
        return {}

    def upnp_SetAVTransportURI(self, *args, **kwargs):
        InstanceID = int(kwargs['InstanceID'])
        CurrentURI = kwargs['CurrentURI']
        CurrentURIMetaData = kwargs['CurrentURIMetaData']
        local_protocol_infos = self.server.connection_manager_server.get_variable('SinkProtocolInfo').value.split(',')
        #print '>>>', local_protocol_infos
        if CurrentURI.startswith('dlna-playcontainer://'):
            def handle_result(r):
                self.load(r[0], r[1])
                return {}

            def pass_error(r):
                return r

            d = defer.maybeDeferred(self.playcontainer_browse, CurrentURI)
            d.addCallback(handle_result)
            d.addErrback(pass_error)
            return d
        elif len(CurrentURIMetaData) == 0:
            self.load(CurrentURI, CurrentURIMetaData)
            return {}
        else:
            elt = DIDLLite.DIDLElement.fromString(CurrentURIMetaData)
            #import pdb; pdb.set_trace()
            if elt.numItems() == 1:
                item = elt.getItems()[0]
                res = item.res.get_matching(local_protocol_infos)
                if len(res) > 0:
                    res = res[0]
                    remote_protocol, remote_network, remote_content_format, _ = res.protocolInfo.split(':')
                    self.load(res.data, CurrentURIMetaData)
                    return {}
        return failure.Failure(errorCode(714))

    def upnp_SetPlayMode(self, *args, **kwargs):
        InstanceID = int(kwargs['InstanceID'])
        NewPlayMode = kwargs['NewPlayMode']

        supported_play_modes = self.server.av_transport_server.get_variable('CurrentPlayMode', instance=InstanceID).allowed_values
        if NewPlayMode not in supported_play_modes:
            return failure.Failure(errorCode(712))
        if self.server:
            self.server.av_transport_server.set_variable(InstanceID, 'CurrentPlayMode', NewPlayMode)
        return {}

    def upnp_X_COHERENCE_SetDisplayTime(self, *args, **kwargs):
        InstanceID = int(kwargs['InstanceID'])
        NewDisplayTime = int(kwargs['NewDisplayTime'])
        if 0 <= NewDisplayTime <= 65535:
            self.display_time = NewDisplayTime
            try:
                self.display_loop.delay(self.display_time)
            except:
                pass
            if self.server:
                self.server.av_transport_server.set_variable(InstanceID, 'X_COHERENCE_DisplayTime', self.display_time)
            return {}
        return failure.Failure(errorCode(601))

    def upnp_X_COHERENCE_SetDisplayTransition(self, *args, **kwargs):
        InstanceID = int(kwargs['InstanceID'])
        NewDisplayTransition = kwargs['NewDisplayTransition']
        supported_transition_modes = self.server.av_transport_server.get_variable('X_COHERENCE_DisplayTransition', instance=InstanceID).allowed_values
        if NewDisplayTransition in supported_transition_modes:
            self.display_transition = NewDisplayTransition
            self.controller.set_transition(NewDisplayTransition)
            if self.server:
                self.server.av_transport_server.set_variable(InstanceID, 'X_COHERENCE_DisplayTransition', self.display_transition)
            return {}
        return failure.Failure(errorCode(600))

    def upnp_init(self):
        self.current_connection_id = None
        self.server.connection_manager_server.set_variable(0, 'SinkProtocolInfo',
                            ['http-get:*:image/jpeg:*',
                             'http-get:*:image/png:*',
                             'http-get:*:image/gif:*'],
                            default=True)

        self.server.av_transport_server.register_vendor_variable('X_COHERENCE_DisplayTime',
                                                                 evented='yes',
                                                                 data_type='ui2', default_value=self.display_time,
                                                                 allowed_value_range={'minumum': 0, 'maximum': 65535, 'step': 1})
        self.server.av_transport_server.register_vendor_action('X_COHERENCE_SetDisplayTime', 'optional',
                                                               (('InstanceID', 'in', 'A_ARG_TYPE_InstanceID'), ('NewDisplayTime', 'in', 'X_COHERENCE_DisplayTime')))
        self.server.av_transport_server.register_vendor_action('X_COHERENCE_GetDisplayTime', 'optional',
                                                               (('InstanceID', 'in', 'A_ARG_TYPE_InstanceID'), ('CurrentDisplayTime', 'out', 'X_COHERENCE_DisplayTime')),
                                                               needs_callback=False)

        self.server.av_transport_server.register_vendor_variable('X_COHERENCE_DisplayTransition',
                                                                 evented='yes',
                                                                 data_type='string', default_value=self.display_transition,
                                                                 allowed_values=self.controller.get_available_transitions())
        self.server.av_transport_server.register_vendor_action('X_COHERENCE_SetDisplayTransition', 'optional',
                                                               (('InstanceID', 'in', 'A_ARG_TYPE_InstanceID'), ('NewDisplayTransition', 'in', 'X_COHERENCE_DisplayTransition')))
        self.server.av_transport_server.register_vendor_action('X_COHERENCE_GetDisplayTransition', 'optional',
                                                               (('InstanceID', 'in', 'A_ARG_TYPE_InstanceID'), ('CurrentDisplayTransition', 'out', 'X_COHERENCE_DisplayTransition')),
                                                               needs_callback=False)

        self.server.av_transport_server.set_variable(0, 'TransportState', 'NO_MEDIA_PRESENT', default=True)
        self.server.av_transport_server.set_variable(0, 'TransportStatus', 'OK', default=True)
        self.server.av_transport_server.set_variable(0, 'CurrentPlayMode', 'REPEAT_ALL', default=True)
        self.server.av_transport_server.set_variable(0, 'CurrentTransportActions', '', default=True)
