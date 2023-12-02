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

from coherence.upnp.core import DIDLLite
from coherence.extern.et import ET
from coherence.upnp.core.utils import parse_xml

from mirabeau.maemo import dialogs, media_renderer

_ = gettext.gettext

class MediaServerBrowser(hildon.StackableWindow):

    def __init__(self, coherence, device):
        super(MediaServerBrowser, self).__init__()
        self.set_title(device.get_friendly_name())

        browse_view = MediaServerBrowseView(self, coherence, device)
        #browse_view.show()
        area = hildon.PannableArea()
        area.add(browse_view)
        area.show_all()
        self.add(area)

class MediaServerBrowseView(hildon.GtkTreeView):

    MS_NAME_COLUMN = 0
    MS_NODE_ID_COLUMN = 1
    MS_UPNP_CLASS_COLUMN = 2
    MS_CHILD_COUNT_COLUMN = 3
    MS_UDN_COLUMN = 4
    MS_SERVICE_PATH_COLUMN = 5
    MS_ICON_COLUMN = 6
    MS_DIDL_COLUMN = 7
    MS_TOOLTIP_COLUMN = 8

    def __init__(self, window, coherence, device):
        hildon.GtkTreeView.__init__(self,gtk.HILDON_UI_MODE_NORMAL)
        self._window = window
        self.coherence = coherence
        self.device = device
        model = gtk.TreeStore(str, str, str, int, str, str, gtk.gdk.Pixbuf,
                              str, gtk.gdk.Pixbuf)
        self.set_model(model)
        column = gtk.TreeViewColumn('Items')
        icon_cell = gtk.CellRendererPixbuf()
        text_cell = gtk.CellRendererText()

        column.pack_start(icon_cell, False)
        column.pack_start(text_cell, True)

        column.set_attributes(text_cell, text=self.MS_NAME_COLUMN)
        column.add_attribute(icon_cell, "pixbuf", self.MS_ICON_COLUMN)
        self.append_column(column)

        self.connect("row-expanded", self.row_got_expanded)
        self.connect("row-collapsed", self.row_got_collapsed)
        self.connect("row-activated", self.row_got_activated)

        icon = resource_filename('mirabeau', os.path.join('data', 'icons','folder.png'))
        self.folder_icon = gtk.gdk.pixbuf_new_from_file(icon)
        icon = resource_filename('mirabeau', os.path.join('data', 'icons','audio-x-generic.png'))
        self.audio_icon = gtk.gdk.pixbuf_new_from_file(icon)
        icon = resource_filename('mirabeau', os.path.join('data', 'icons','video-x-generic.png'))
        self.video_icon = gtk.gdk.pixbuf_new_from_file(icon)
        icon = resource_filename('mirabeau', os.path.join('data', 'icons','image-x-generic.png'))
        self.image_icon = gtk.gdk.pixbuf_new_from_file(icon)

        self.load()

    def load(self):
        service = self.device.get_service_by_type('ContentDirectory')
        service.subscribe_for_variable('ContainerUpdateIDs', callback=self.state_variable_change)
        service.subscribe_for_variable('SystemUpdateID', callback=self.state_variable_change)

        model = self.get_model()
        #item = model.append(None)
        #model.set_value(item, self.MS_ICON_COLUMN, self.folder_icon)
        #model.set_value(item, self.MS_NAME_COLUMN, 'root')
        #model.set_value(item, self.MS_NODE_ID_COLUMN, '0')
        self.browse_object('0')

    def state_variable_change( self, variable):
        name = variable.name
        value = variable.value
        if name == 'ContainerUpdateIDs':
            changes = value.split(',')
            model = self.get_model()

            while len(changes) > 1:
                container = changes.pop(0).strip()
                update_id = changes.pop(0).strip()

                def match_func(iter, data):
                    column, key = data # data is a tuple containing column number, key
                    value = model.get_value(iter, column)
                    return value == key

                def search(iter, func, data):
                    while iter:
                        if func(iter, data):
                            return iter
                        result = search(model.iter_children(iter), func, data)
                        if result: return result
                        iter = model.iter_next(iter)
                    return None

                row_count = 0
                for row in model:
                    iter = model.get_iter(row_count)
                    match_iter = search(model.iter_children(iter),
                                        match_func, (self.MS_NODE_ID_COLUMN, container))
                    if match_iter:
                        path = model.get_path(match_iter)
                        expanded = self.row_expanded(path)
                        child = model.iter_children(match_iter)
                        while child:
                            model.remove(child)
                            child = model.iter_children(match_iter)
                        self.browse(self.treeview, path, None,
                                    starting_index=0, requested_count=0,
                                    force=True, expand=expanded)

                    break
                    row_count += 1

    def row_got_expanded(self, view, iter, row_path):
        model = self.get_model()
        child = model.iter_children(iter)
        if child:
            upnp_class = model.get(child, self.MS_UPNP_CLASS_COLUMN)[0]
            if upnp_class == 'placeholder':
                self.browse(view, row_path, None)

    def row_got_collapsed(self, view, iter, row_path):
        model = self.get_model()

    def row_got_activated(self, view, row_path, column):
        model = self.get_model()
        iter = model.get_iter(row_path)
        upnp_class = model.get(iter, self.MS_UPNP_CLASS_COLUMN)[0]
        if upnp_class.startswith('object.container'):
            if view.row_expanded(row_path):
                view.collapse_row(row_path)
            return

        # FIXME, only when we have a Resource item, then play it
        # this is a leaf, let's play it
        didl_fragment = model.get(iter, self.MS_DIDL_COLUMN)[0]
        url = model.get(iter, self.MS_SERVICE_PATH_COLUMN)[0]
        if url == '':
            return
        dialog = dialogs.SelectMRDialog(self._window, self.coherence)
        response = dialog.run()
        if response == gtk.RESPONSE_ACCEPT:
            media_render_device = dialog.get_media_renderer()
            window = media_renderer.MediaRendererWindow(self.coherence,
                                                        media_render_device)
            window.load_media(didl_fragment, url)
            window.show_all()

        dialog.destroy()

    def browse(self, view, row_path, column, starting_index=0, requested_count=0,
               force=False, expand=False):
        model = self.get_model()
        iter = model.get_iter(row_path)
        child = model.iter_children(iter)
        if child:
            upnp_class = model.get(child, self.MS_UPNP_CLASS_COLUMN)[0]
            if upnp_class != 'placeholder':
                if force == False:
                    if view.row_expanded(row_path):
                        view.collapse_row(row_path)
                    else:
                        view.expand_row(row_path, False)
                    return

        object_id = model.get(iter, self.MS_NODE_ID_COLUMN)[0]
        self.browse_object(object_id, iter=iter, view=view, row_path=row_path,
                              starting_index=starting_index,
                              requested_count=requested_count,
                              force=force, expand=expand)

    def browse_object(self, object_id, iter=None, view=None, row_path=None,
                      starting_index=0, requested_count=0,
                      force=False, expand=False):
        model = self.get_model()

        def reply(r,service):
            if iter:
                child = model.iter_children(iter)
                if child:
                    upnp_class = model.get(child, self.MS_UPNP_CLASS_COLUMN)[0]
                    if upnp_class == 'placeholder':
                        model.remove(child)

                title = model.get(iter, self.MS_NAME_COLUMN)[0]
                if title:
                    try:
                        title = title[:title.rindex('(')]
                        model.set_value(iter, self.MS_NAME_COLUMN,
                                        "%s(%d)" % (title, int(r['TotalMatches'])))
                    except ValueError:
                        pass
            elt = parse_xml(r['Result'], 'utf-8')
            elt = elt.getroot()
            for child in elt:
                stored_didl_string = DIDLLite.element_to_didl(ET.tostring(child))
                didl = DIDLLite.DIDLElement.fromString(stored_didl_string)
                item = didl.getItems()[0]
                if item.upnp_class.startswith('object.container'):
                    icon = self.folder_icon
                    #service = model.get(iter, self.MS_SERVICE_PATH_COLUMN)[0]
                    child_count = item.childCount
                    try:
                        title = "%s (%d)" % (item.title, item.childCount)
                    except TypeError:
                        title = "%s (n/a)" % item.title
                        child_count = -1
                else:
                    icon = None
                    service = ''

                    child_count = -1
                    title = item.title
                    if item.upnp_class.startswith('object.item.audioItem'):
                        icon = self.audio_icon
                    elif item.upnp_class.startswith('object.item.videoItem'):
                        icon = self.video_icon
                    elif item.upnp_class.startswith('object.item.imageItem'):
                        icon = self.image_icon

                    res = item.res.get_matching(['*:*:*:*'], protocol_type='http-get')
                    if len(res) > 0:
                        res = res[0]
                        service = res.data

                new_iter = model.append(iter, (title, item.id, item.upnp_class, child_count,
                                               '',service,icon,stored_didl_string,None))
                if item.upnp_class.startswith('object.container'):
                    model.append(new_iter, ('...loading...',
                                            '', 'placeholder', -1, '', '',
                                            None, '', None))


            if ((int(r['TotalMatches']) > 0 and force==False) or \
                expand==True):
                if view:
                    view.expand_row(row_path, False)

            if(requested_count != int(r['NumberReturned']) and \
               int(r['NumberReturned']) < (int(r['TotalMatches'])-starting_index)):
                self.browse(view, row_path, column,
                            starting_index=starting_index+int(r['NumberReturned']),
                            force=True)

        service = self.device.get_service_by_type('ContentDirectory')
        action = service.get_action('Browse')
        d = action.call(ObjectID=object_id,BrowseFlag='BrowseDirectChildren',
                        StartingIndex=str(starting_index),RequestedCount=str(requested_count),
                        Filter='*',SortCriteria='')
        d.addCallback(reply,service)
        d.addErrback(self.handle_error)
        return d

    def handle_error(self,error):
        print error
