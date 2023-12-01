# -*- coding: utf-8 -*-
#
# Licensed under the MIT license
# http://opensource.org/licenses/mit-license.php

# Copyright 2010 - Philippe Normand <phil@base-art.net>
#                  Frank Scholz <coherence@beebits.net>

import os
import gettext

import hildon
import pygtk
pygtk.require('2.0')
import gtk

from pkg_resources import resource_filename

from twisted.internet import task

from coherence.upnp.core.utils import parse_xml, getPage, means_true
from coherence.upnp.core import DIDLLite

_ = gettext.gettext

class MediaRendererWindow(hildon.StackableWindow):

    def __init__(self, coherence, device):
        super(MediaRendererWindow, self).__init__()
        self.coherence = coherence
        self.device = device
        self.set_title(device.get_friendly_name())

        vbox = gtk.VBox(homogeneous=False, spacing=10)
        vbox.set_border_width(5)
        
        hbox = gtk.HBox(homogeneous=False)
        self.album_art_image = gtk.Image()
        icon = '/usr/share/icons/hicolor/295x295/hildon/mediaplayer_default_album.png'
        self.blank_icon = gtk.gdk.pixbuf_new_from_file(icon)
        self.album_art_image.set_from_pixbuf(self.blank_icon)
        hbox.pack_start(self.album_art_image)#,False,False,2)

        #icon_loader = gtk.gdk.PixbufLoader()
        #icon_loader.write(urllib.urlopen(str(res.data)).read())
        #icon_loader.close()

        rightbox = gtk.VBox(homogeneous=False, spacing=10)

        textbox = gtk.VBox(homogeneous=False, spacing=10)
        self.title_text = gtk.Label("<b>title</b>")
        self.title_text.set_use_markup(True)
        textbox.pack_start(self.title_text,False,False,2)
        self.album_text = gtk.Label("album")
        self.album_text.set_use_markup(True)
        textbox.pack_start(self.album_text,False,False,2)
        self.artist_text = gtk.Label("artist")
        self.artist_text.set_use_markup(True)
        textbox.pack_start(self.artist_text,False,False,2)
        rightbox.pack_start(textbox,False,False,2)

        seekbox = gtk.HBox(homogeneous=False, spacing=10)
        self.position_min_text = gtk.Label("0:00")
        self.position_min_text.set_use_markup(True)
        seekbox.pack_start(self.position_min_text,False,False,2)
        adjustment=gtk.Adjustment(value=0, lower=0, upper=240, step_incr=1,page_incr=20)
        self.position_scale = gtk.HScale(adjustment=adjustment)
        self.position_scale.set_draw_value(False)
        self.position_scale.set_sensitive(False)
        self.position_scale.connect("format-value", self.format_position)
        self.position_scale.connect('change-value',self.position_changed)
        seekbox.pack_start(self.position_scale,True,True,2)
        self.position_max_text = gtk.Label("0:00")
        self.position_max_text.set_use_markup(True)
        seekbox.pack_end(self.position_max_text,False,False,2)
        rightbox.pack_start(seekbox,False,False,2)

        hbox.pack_start(rightbox)

        vbox.pack_start(hbox,False,False,2)

        
        self.prev_button = self.make_button('media-skip-backward.png',
                                            self.skip_backward,sensitive=False)
        self.seek_backward_button = self.make_button('media-seek-backward.png',
                                                     callback=self.seek_backward,sensitive=False)
        self.stop_button = self.make_button('media-playback-stop.png',
                                            callback=self.stop,sensitive=False)
        self.start_button = self.make_button('media-playback-start.png',
                                             callback=self.play_or_pause,sensitive=False)
        self.seek_forward_button = self.make_button('media-seek-forward.png',
                                                    callback=self.seek_forward,sensitive=False)
        self.next_button = self.make_button('media-skip-forward.png',
                                            self.skip_forward,sensitive=False)
        buttonbox2 = gtk.HBox(homogeneous=True, spacing=30)
        buttonbox2.pack_start(self.prev_button,False,False,2)
        buttonbox2.pack_start(self.seek_backward_button,False,False,2)
        buttonbox2.pack_start(self.stop_button,False,False,2)
        buttonbox2.pack_start(self.start_button,False,False,2)
        buttonbox2.pack_start(self.seek_forward_button,False,False,2)
        buttonbox2.pack_start(self.next_button,False,False,2)

        buttonbox = gtk.HBox(homogeneous=False, spacing=10)
        self.buttons_alignment = gtk.Alignment(0.25, 0.25, 0.25, 0.25)
        self.buttons_alignment.add(buttonbox2)
        buttonbox.pack_start(self.buttons_alignment,False,False,2)

        
        hbox = gtk.HBox(homogeneous=False, spacing=10)
        #hbox.set_size_request(240,-1)
        adjustment=gtk.Adjustment(value=0, lower=0, upper=100, step_incr=1,page_incr=20)#, page_size=20)
        self.volume_scale = gtk.HScale(adjustment=adjustment)
        self.volume_scale.set_size_request(700,-1)
        self.volume_scale.set_draw_value(False)
        self.volume_scale.connect('change-value',self.volume_changed)
        #hbox.pack_start(self.volume_scale,False,False,2)
        button = hildon.GtkButton(gtk.HILDON_SIZE_FINGER_HEIGHT)
        self.volume_image = gtk.Image()
        icon = resource_filename('mirabeau', os.path.join('data', 'icons','audio-volume-low.png'))
        self.volume_low_icon = gtk.gdk.pixbuf_new_from_file(icon)
        self.volume_image.set_from_pixbuf(self.volume_low_icon)
        button.set_image(self.volume_image)
        #button.connect("clicked", self.mute)
        button.connect("clicked", self.volume_toggle)

        self.volume_alignment = gtk.Alignment(0.25, 0.25, 1, 0.25)
        self.volume_alignment.add(self.volume_scale)
        
        icon = resource_filename('mirabeau', os.path.join('data', 'icons','audio-volume-medium.png'))
        self.volume_medium_icon = gtk.gdk.pixbuf_new_from_file(icon)
        icon = resource_filename('mirabeau', os.path.join('data', 'icons','audio-volume-high.png'))
        self.volume_high_icon = gtk.gdk.pixbuf_new_from_file(icon)
        icon = resource_filename('mirabeau', os.path.join('data', 'icons','audio-volume-muted.png'))
        self.volume_muted_icon = gtk.gdk.pixbuf_new_from_file(icon)
        hbox.pack_end(button,False,False,2)

        buttonbox.pack_end(hbox,False,False,2)
        vbox.pack_end(buttonbox, False,False,2)
        #vbox.pack_start(self.buttons_alignment,False,False,2)

        self.pause_button_image = gtk.Image()
        icon = resource_filename('mirabeau', os.path.join('data', 'icons','media-playback-pause.png'))
        icon = gtk.gdk.pixbuf_new_from_file(icon)
        self.pause_button_image.set_from_pixbuf(icon)
        self.start_button_image = self.start_button.get_image()

        self.status_bar = gtk.Statusbar()
        context_id = self.status_bar.get_context_id("Statusbar")
        #vbox.pack_end(self.status_bar,False,False,2)

        self.add(vbox)
        self.vbox = vbox
        self.buttonbox = buttonbox

        self.seeking = False

        self.position_loop = task.LoopingCall(self.get_position)

        service = self.device.get_service_by_type('RenderingControl')
        service.subscribe_for_variable('Volume', callback=self.state_variable_change)
        service.subscribe_for_variable('Mute', callback=self.state_variable_change)

        service = self.device.get_service_by_type('AVTransport')
        if service != None:
            service.subscribe_for_variable('AVTransportURI', callback=self.state_variable_change)
            service.subscribe_for_variable('CurrentTrackMetaData', callback=self.state_variable_change)
            service.subscribe_for_variable('TransportState', callback=self.state_variable_change)
            service.subscribe_for_variable('CurrentTransportActions', callback=self.state_variable_change)

            service.subscribe_for_variable('AbsTime', callback=self.state_variable_change)
            service.subscribe_for_variable('TrackDuration', callback=self.state_variable_change)

        self.get_position()
        self.connect("delete-event", self.clean)

    def clean(self, widget, event):
        if self.position_loop.running:
            self.position_loop.stop()

    def volume_toggle(self, widget):
        if self.volume_alignment.props.visible:
            self.buttonbox.remove(self.volume_alignment)
            self.buttonbox.pack_start(self.buttons_alignment, False, False, 2)
            self.buttons_alignment.show_all()
            self.volume_alignment.hide_all()
        else:
            self.buttonbox.remove(self.buttons_alignment)
            self.buttonbox.pack_start(self.volume_alignment,False,False,2)
            self.volume_alignment.show_all()
            self.buttons_alignment.hide_all()

    def make_button(self,icon,callback=None,sensitive=True):
        icon = resource_filename('mirabeau', os.path.join('data', 'icons',icon))
        icon = gtk.gdk.pixbuf_new_from_file(icon)
        button = hildon.GtkButton(gtk.HILDON_SIZE_FINGER_HEIGHT)
        image = gtk.Image()
        image.set_from_pixbuf(icon)
        button.set_image(image)
        button.connect("clicked", lambda x: callback())
        button.set_sensitive(sensitive)
        return button

    def load_media(self, metadata, url):
        elt = DIDLLite.DIDLElement.fromString(metadata)
        if elt.numItems() == 1:
            service = self.device.get_service_by_type('ConnectionManager')
            if service != None:
                local_protocol_infos=service.get_state_variable('SinkProtocolInfo').value.split(',')
                item = elt.getItems()[0]
                try:
                    res = item.res.get_matching(local_protocol_infos, protocol_type='internal')
                    if len(res) == 0:
                        res = item.res.get_matching(local_protocol_infos)
                    if len(res) > 0:
                        res = res[0]
                        remote_protocol,remote_network,remote_content_format,_ = res.protocolInfo.split(':')
                        d = self.stop()
                        d.addCallback(lambda x: self.set_uri(res.data,metadata))
                        d.addCallback(lambda x: self.play_or_pause(force_play=True))
                        d.addErrback(self.handle_error)
                        d.addErrback(self.handle_error)
                except AttributeError:
                    print "Sorry, we currently support only single items!"
            else:
                print "can't check for the best resource!"

    def state_variable_change(self,variable):
        if variable.name == 'CurrentTrackMetaData':
            if variable.value != None and len(variable.value)>0:
                try:
                    elt = DIDLLite.DIDLElement.fromString(variable.value)
                except SyntaxError:
                    print "seems we haven't got an XML string", repr(variable.value)
                else:
                    for item in elt.getItems():
                        self.title_text.set_markup("<b>%s</b>" % item.title)
                        if item.album != None:
                            self.album_text.set_markup(item.album)
                        else:
                            self.album_text.set_markup('')
                        if item.artist != None:
                            self.artist_text.set_markup("<i>%s</i>" % item.artist)
                        else:
                            self.artist_text.set_markup("")

                        def got_icon(icon):
                            icon = icon[0]
                            icon_loader = gtk.gdk.PixbufLoader()
                            icon_loader.write(icon)
                            icon_loader.close()
                            icon = icon_loader.get_pixbuf()
                            icon = icon.scale_simple(300, 300, gtk.gdk.INTERP_BILINEAR)
                            self.album_art_image.set_from_pixbuf(icon)

                        if item.upnp_class.startswith('object.item.audioItem') and item.albumArtURI != None:
                            d = getPage(item.albumArtURI)
                            d.addCallback(got_icon)
                        elif item.upnp_class.startswith('object.item.imageItem'):
                            res = item.res.get_matching('http-get:*:image/:*')
                            if len(res) > 0:
                                res = res[0]
                                d = getPage(res.data)
                                d.addCallback(got_icon)
                            else:
                                self.album_art_image.set_from_pixbuf(self.blank_icon)
                        else:
                            self.album_art_image.set_from_pixbuf(self.blank_icon)
            else:
                self.title_text.set_markup('')
                self.album_text.set_markup('')
                self.artist_text.set_markup('')
                self.album_art_image.set_from_pixbuf(self.blank_icon)

        elif variable.name == 'TransportState':
            if variable.value == 'PLAYING':
                service = self.device.get_service_by_type('AVTransport')
                if 'Pause' in service.get_actions():
                    self.start_button.set_image(self.pause_button_image)
                try:
                    self.position_loop.start(1.0, now=True)
                except:
                    pass
            elif variable.value != 'TRANSITIONING':
                self.start_button.set_image(self.start_button_image)
                try:
                    self.position_loop.stop()
                except:
                    pass
            if variable.value == 'STOPPED':
                self.get_position()


            context_id = self.status_bar.get_context_id("Statusbar")
            self.status_bar.pop(context_id)
            self.status_bar.push(context_id,"%s" % variable.value)

        elif variable.name == 'CurrentTransportActions':
            try:
                actions = map(lambda x: x.upper(),variable.value.split(','))
                if 'SEEK' in actions:
                    self.position_scale.set_sensitive(True)
                    self.seek_forward_button.set_sensitive(True)
                    self.seek_backward_button.set_sensitive(True)
                else:
                    self.position_scale.set_sensitive(False)
                    self.seek_forward_button.set_sensitive(False)
                    self.seek_backward_button.set_sensitive(False)
                self.start_button.set_sensitive('PLAY' in actions)
                self.stop_button.set_sensitive('STOP' in actions)
                self.prev_button.set_sensitive('PREVIOUS' in actions)
                self.next_button.set_sensitive('NEXT' in actions)
            except:
                #very unlikely to happen
                import traceback
                print traceback.format_exc()

        elif variable.name == 'AVTransportURI':
            if variable.value != '':
                pass
                #self.seek_backward_button.set_sensitive(True)
                #self.stop_button.set_sensitive(True)
                #self.start_button.set_sensitive(True)
                #self.seek_forward_button.set_sensitive(True)
            else:
                #self.seek_backward_button.set_sensitive(False)
                #self.stop_button.set_sensitive(False)
                #self.start_button.set_sensitive(False)
                #self.seek_forward_button.set_sensitive(False)
                self.album_art_image.set_from_pixbuf(self.blank_icon)
                self.title_text.set_markup('')
                self.album_text.set_markup('')
                self.artist_text.set_markup('')

        elif variable.name == 'Volume':
            try:
                volume = int(variable.value)
                if int(self.volume_scale.get_value()) != volume:
                    self.volume_scale.set_value(volume)
                service = self.device.get_service_by_type('RenderingControl')
                mute_variable = service.get_state_variable('Mute')
                if means_true(mute_variable.value) == True:
                    self.volume_image.set_from_pixbuf(self.volume_muted_icon)
                elif volume < 34:
                    self.volume_image.set_from_pixbuf(self.volume_low_icon)
                elif volume < 67:
                    self.volume_image.set_from_pixbuf(self.volume_medium_icon)
                else:
                    self.volume_image.set_from_pixbuf(self.volume_high_icon)

            except:
                import traceback
                print traceback.format_exc()

        elif variable.name == 'Mute':
            service = self.device.get_service_by_type('RenderingControl')
            volume_variable = service.get_state_variable('Volume')
            volume = volume_variable.value
            if means_true(variable.value) == True:
                self.volume_image.set_from_pixbuf(self.volume_muted_icon)
            elif volume < 34:
                self.volume_image.set_from_pixbuf(self.volume_low_icon)
            elif volume < 67:
                self.volume_image.set_from_pixbuf(self.volume_medium_icon)
            else:
                self.volume_image.set_from_pixbuf(self.volume_high_icon)

    def seek_backward(self):
        self.seeking = True
        value = self.position_scale.get_value()
        value = int(value)
        seconds = max(0,value-20)

        hours = seconds / 3600
        seconds = seconds - hours * 3600
        minutes = seconds / 60
        seconds = seconds - minutes * 60
        target = "%d:%02d:%02d" % (hours,minutes,seconds)

        def handle_result(r):
            self.seeking = False
            #self.get_position()

        service = self.device.get_service_by_type('AVTransport')
        seek_modes = service.get_state_variable('A_ARG_TYPE_SeekMode').allowed_values
        unit = 'ABS_TIME'
        if 'ABS_TIME' not in seek_modes:
            if 'REL_TIME' in seek_modes:
                unit = 'REL_TIME'
                target = "-%d:%02d:%02d" % (0,0,20)

        action = service.get_action('Seek')
        d = action.call(InstanceID=0,Unit=unit,Target=target)
        d.addCallback(handle_result)
        d.addErrback(self.handle_error)
        return d

    def seek_forward(self):
        self.seeking = True
        value = self.position_scale.get_value()
        value = int(value)
        max = int(self.position_scale.get_adjustment().upper)
        seconds = min(max,value+20)

        hours = seconds / 3600
        seconds = seconds - hours * 3600
        minutes = seconds / 60
        seconds = seconds - minutes * 60
        target = "%d:%02d:%02d" % (hours,minutes,seconds)

        def handle_result(r):
            self.seeking = False
            #self.get_position()

        service = self.device.get_service_by_type('AVTransport')
        seek_modes = service.get_state_variable('A_ARG_TYPE_SeekMode').allowed_values
        unit = 'ABS_TIME'
        if 'ABS_TIME' not in seek_modes:
            if 'REL_TIME' in seek_modes:
                unit = 'REL_TIME'
                target = "+%d:%02d:%02d" % (0,0,20)

        action = service.get_action('Seek')
        d = action.call(InstanceID=0,Unit=unit,Target=target)
        d.addCallback(handle_result)
        d.addErrback(self.handle_error)
        return d

    def play_or_pause(self,force_play=False):
        service = self.device.get_service_by_type('AVTransport')
        variable = service.get_state_variable('TransportState', instance=0)
        if force_play == True or variable.value != 'PLAYING':
            action = service.get_action('Play')
            d = action.call(InstanceID=0,Speed=1)
        else:
            action = service.get_action('Pause')
            d = action.call(InstanceID=0)
        d.addCallback(self.handle_result)
        d.addErrback(self.handle_error)
        return d

    def stop(self):
        service = self.device.get_service_by_type('AVTransport')
        action = service.get_action('Stop')
        d = action.call(InstanceID=0)
        d.addCallback(self.handle_result)
        d.addErrback(self.handle_error)
        return d

    def skip_backward(self):
        service = self.device.get_service_by_type('AVTransport')
        action = service.get_action('Previous')
        d = action.call(InstanceID=0)
        d.addCallback(self.handle_result)
        d.addErrback(self.handle_error)
        return d

    def skip_forward(self):
        service = self.device.get_service_by_type('AVTransport')
        action = service.get_action('Next')
        d = action.call(InstanceID=0)
        d.addCallback(self.handle_result)
        d.addErrback(self.handle_error)
        return d

    def set_uri(self, uri, didl):
        service = self.device.get_service_by_type('AVTransport')
        action = service.get_action('SetAVTransportURI')
        d = action.call(InstanceID=0,CurrentURI=uri,
                                     CurrentURIMetaData=didl)
        d.addCallback(self.handle_result)
        d.addErrback(self.handle_error)
        return d


    def position_changed(self,range,scroll,value):

        old_value = self.position_scale.get_value()
        new_value = value - old_value
        if new_value < 0 and new_value > -1.0:
            return
        if new_value >= 0 and new_value < 1.0:
            return

        self.seeking = True
        adjustment = range.get_adjustment()
        value = int(value)
        max = int(adjustment.upper)
        seconds = target_seconds = min(max,value)

        hours = seconds / 3600
        seconds = seconds - hours * 3600
        minutes = seconds / 60
        seconds = seconds - minutes * 60
        target = "%d:%02d:%02d" % (hours,minutes,seconds)

        service = self.device.get_service_by_type('AVTransport')

        seek_modes = service.get_state_variable('A_ARG_TYPE_SeekMode').allowed_values
        unit = 'ABS_TIME'
        if 'ABS_TIME' not in seek_modes:
            if 'REL_TIME' in seek_modes:
                unit = 'REL_TIME'
                seconds = int(new_value)

                sign = '+'
                if seconds < 0:
                    sign = '-'
                    seconds = seconds * -1

                hours = seconds / 3600
                seconds = seconds - hours * 3600
                minutes = seconds / 60
                seconds = seconds - minutes * 60
                target = "%s%d:%02d:%02d" % (sign,hours,minutes,seconds)

        def handle_result(r):
            self.seeking = False
            #self.get_position()

        action = service.get_action('Seek')
        d = action.call(InstanceID=0,Unit=unit,Target=target)
        d.addCallback(handle_result)
        d.addErrback(self.handle_error)

    def format_position(self,scale,value):
        seconds = int(value)
        hours = seconds / 3600
        seconds = seconds - hours * 3600
        minutes = seconds / 60
        seconds = seconds - minutes * 60
        if hours > 0:
            return "%d:%02d:%02d" % (hours,minutes,seconds)
        else:
            return "%d:%02d" % (minutes,seconds)

    def get_position(self):

        if self.seeking == True:
            return

        def handle_result(r,service):
            try:
                duration = r['TrackDuration']
                h,m,s = duration.split(':')
                if int(h) > 0:
                    duration = '%d:%02d:%02d' % (int(h),int(m),int(s))
                else:
                    duration = '%d:%02d' % (int(m),int(s))
                max = (int(h) * 3600) + (int(m)*60) + int(s)
                if max > 0:
                    self.position_scale.set_range(0,max)
                self.position_max_text.set_markup(duration)
                actions = service.get_state_variable('CurrentTransportActions')
                try:
                    actions = actions.value.split(',')
                    if 'SEEK' in actions:
                        self.position_scale.set_sensitive(True)
                except AttributeError:
                    pass
            except:
                #import traceback
                try:
                    self.position_scale.set_range(0,0)
                except:
                    pass
                self.position_max_text.set_markup('0:00')
                self.position_scale.set_sensitive(False)

            try:
                if self.seeking == False:
                    position = r['AbsTime']
                    h,m,s = position.split(':')
                    position = (int(h) * 3600) + (int(m)*60) + int(s)
                    self.position_scale.set_value(position)
            except:
                #import traceback
                #print traceback.format_exc()
                pass

        service = self.device.get_service_by_type('AVTransport')
        try:
            action = service.get_action('GetPositionInfo')
            d = action.call(InstanceID=0)
            d.addCallback(handle_result,service)
            d.addErrback(self.handle_error)
            return d
        except AttributeError:
            # the device and its services are gone
            pass

    def volume_changed(self,range,scroll,value):
        value = int(value)
        if value > 100:
            value = 100
        service = self.device.get_service_by_type('RenderingControl')
        action = service.get_action('SetVolume')
        d = action.call(InstanceID=0,
                    Channel='Master',
                    DesiredVolume=value)
        d.addCallback(self.handle_result)
        d.addErrback(self.handle_error)
        return d

    def mute(self,w):
        service = self.device.get_service_by_type('RenderingControl')
        action = service.get_action('SetMute')
        mute_variable = service.get_state_variable('Mute')
        if means_true(mute_variable.value) == False:
            new_mute = '1'
        else:
            new_mute = '0'
        d = action.call(InstanceID=0,
                        Channel='Master',
                        DesiredMute=new_mute)
        d.addCallback(self.handle_result)
        d.addErrback(self.handle_error)
        return d

    def handle_error(self,e):
        print 'we have an error', e
        return e

    def handle_result(self,r):
        return r
