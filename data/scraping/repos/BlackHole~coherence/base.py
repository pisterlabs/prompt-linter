# Licensed under the MIT license
# http://opensource.org/licenses/mit-license.php

# Copyright 2009 Frank Scholz <coherence@beebits.net>

import os,random

try:
    _ = set()
except:
    from sets import Set as set

import traceback

import mimetypes
mimetypes.init()

# Twisted
from twisted.internet import defer, reactor
from twisted.python.filepath import FilePath


# Coherence
from coherence.base import Coherence
from coherence.upnp.devices.media_renderer import MediaRenderer

import coherence.extern.louie as louie
from coherence import log

from coherence.upnp.core.utils import means_true

from cadre.renderer import CadreRenderer

class Cadre(log.Loggable):

    logCategory = 'renderer'

    def __init__(self,config):
        self.config = config
        fullscreen = False
        try:
            if means_true(self.config['fullscreen']):
                fullscreen = True
        except:
            pass
        grafics = self.config.get('grafics')
        if grafics == 'pyglet':
            from cadre.scribbling.pyglet_backend import Canvas
        else:
            from cadre.scribbling.clutter_backend import Canvas
        self.canvas = Canvas(fullscreen)

        try:
            overlays = self.config['overlay']
            if not isinstance(overlays,list):
                overlays = [overlays]
        except:
            overlays = []

        map(self.canvas.add_overlay,overlays)

        coherence_config = {}
        #coherence_config['version'] = '1'
        coherence_config['logmode'] = 'warning'
        #coherence_config['controlpoint'] = 'yes'

        louie.connect(self.media_server_found, 'Coherence.UPnP.ControlPoint.MediaServer.detected', louie.Any)
        louie.connect(self.media_server_removed, 'Coherence.UPnP.ControlPoint.MediaServer.removed', louie.Any)
        self.coherence = Coherence(coherence_config)


        name = self.config.get('name','Cadre - Coherence Picture-Frame')
        kwargs = {'version':1,
                'no_thread_needed':True,
                'name':name}

        kwargs['transition'] = 'NONE'
        try:
            if self.config['transition'].upper() in self.get_available_transitions():
                kwargs['transition'] = self.config['transition'].upper()
                self.set_transition(self.config['transition'].upper())
        except:
            pass

        try:
            kwargs['display_time'] = int(self.config['display_time'])
        except:
            pass

        self.canvas.set_title(name)

        kwargs['controller'] = self
        uuid = self.config.get('uuid')
        if uuid:
            kwargs['uuid'] = uuid
        print kwargs
        self.renderer = MediaRenderer(self.coherence,CadreRenderer,**kwargs)

        if 'uuid' not in self.config:
            self.config['uuid'] = str(self.renderer.uuid)[5:]
            try:
                self.config.save()
            except AttributeError:
                pass
        reactor.callLater(0.5,self.start,name)

    def init_logging(self):
        logmode = self.config.get('logging').get('level','warning')
        _debug = []

        try:
            subsystems = self.config.get('logging')['subsystem']
        except KeyError:
            subsystems = []

        if isinstance(subsystems,dict):
            subsystems = [subsystems]
        for subsystem in subsystems:
            try:
                if subsystem['active'] == 'no':
                    continue
            except (KeyError,TypeError):
                pass
            self.info( "setting log-level for subsystem %s to %s" % (subsystem['name'],subsystem['level']))
            _debug.append('%s:%d' % (subsystem['name'].lower(), log.human2level(subsystem['level'])))

        if len(_debug) > 0:
            _debug = ','.join(_debug)
        else:
            _debug = '*:%d' % log.human2level(logmode)

        logfile = self.config.get('logging').get('logfile',None)
        if logfile != None:
            logfile = unicode(logfile)

        log.init(logfile, _debug)

    def start(self,name):
        self.canvas.set_title(name)

        try:
            self.content = self.config['content']
        except:
            self.content = []
        if not isinstance( self.content, list):
            self.content = [self.content]

        tmp_l = []
        for path in self.content:
            if path.startswith('http://'):
                tmp_l.append(path)
            else:
                tmp_l.append(os.path.abspath(path))
        self.content = tmp_l

        self.items = []
        self.playlist = []
        self.warning("checking for items...")
        for path in self.content:
            if path.startswith('http://'):
                self.items.append(path)
            else:
                self.walk(path)
        self.warning("done")

        self.renderer.av_transport_server.get_variable('AVTransportURI').subscribe(self.state_variable_change)
        self.renderer.av_transport_server.get_variable('NextAVTransportURI').subscribe(self.state_variable_change)
        #self.renderer.av_transport_server.get_variable('TransportState').subscribe(self.state_variable_change)
        #self.renderer.av_transport_server.get_variable('AVTransportURI').subscribe(self.state_variable_change)
        #self.renderer.av_transport_server.get_variable('LastChange').subscribe(self.state_variable_change)
        try:
            if means_true(self.config['autostart']):
                d = defer.maybeDeferred(self.get_next_item)
                d.addCallback(lambda result: self.set_renderer_uri(self.renderer,result[0],result[1]))
                d.addErrback(self.got_error)

                def get_next():
                    d = defer.maybeDeferred(self.get_next_item)
                    d.addCallback(lambda result: self.set_renderer_next_uri(self.renderer,result[0],result[1]))
                    d.addErrback(self.got_error)

                d.addCallback(lambda result: get_next())
                d.addErrback(self.got_error)

        except KeyError:
            pass
        except:
            traceback.print_exc()

    def quit(self):
        reactor.stop()

    def got_error(self,error):
        self.warning("error %r" % error)
        error.printTraceback()

    def get_available_transitions(self):
        try:
            return self.canvas.get_available_transitions()
        except:
            return ['NONE']

    def set_transition(self,transition):
        if transition in self.get_available_transitions():
            self.canvas.transition = transition

    def get_next_item(self):
        try:
            uri = self.playlist.pop()
        except IndexError:
            self.playlist = self.items[:]
            if means_true(self.config['shuffle']):
                random.shuffle(self.playlist)
            uri = self.playlist.pop()

        try:
            uri = "file://" + uri.path
            return uri,''
        except:
            if uri.startswith('http://'):
                pass
            raise

    def set_renderer_uri(self,renderer,uri,meta=''):
        renderer.backend.stop()
        renderer.backend.load(uri,meta)
        renderer.backend.play()

    def set_renderer_next_uri(self,renderer,uri,meta=''):
        renderer.backend.upnp_SetNextAVTransportURI(InstanceID='0',NextURI=uri,NextURIMetaData=meta)

    def walk(self, path):
        containers = []
        filepath = FilePath(path)
        if filepath.isdir():
            containers.append(filepath)
        elif filepath.isfile():
            self.items.append(FilePath(path))
        while len(containers)>0:
            container = containers.pop()
            try:
                for child in container.children():
                    if child.isdir():
                        containers.append(child)
                    elif child.isfile() or child.islink():
                        mimetype,_ = mimetypes.guess_type(child.path, strict=False)
                        if mimetype and mimetype.startswith("image/"):
                            self.items.append(child)
            except UnicodeDecodeError:
                self.warning("UnicodeDecodeError - there is something wrong with a file located in %r", container.get_path())

    def media_server_found(self, client,udn):
        print "media_server_found", client.device.get_friendly_name()

    def media_server_removed(self, udn):
        print "media_server_removed", udn

    def state_variable_change(self,variable):
        self.warning("state_variable %r changed from %r -> %r" % (variable.name,variable.old_value,variable.value))
        if variable.name == 'NextAVTransportURI':
            if variable.value == '' and self.renderer.av_transport_server.get_variable('TransportState').value == 'TRANSITIONING':
                d = defer.maybeDeferred(self.get_next_item)
                d.addCallback(lambda result: self.set_renderer_next_uri(self.renderer,result[0],result[1]))
                d.addErrback(self.got_error)
