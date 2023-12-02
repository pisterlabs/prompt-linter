# -*- coding: utf-8 -*-
#
# Licensed under the MIT license
# http://opensource.org/licenses/mit-license.php

# Copyright 2010 - Philippe Normand <phil@base-art.net>

import gettext

import hildon
import pygtk
pygtk.require('2.0')
import gtk
import dbus

_ = gettext.gettext

from coherence.extern.telepathy import connect

from telepathy.interfaces import ACCOUNT_MANAGER, ACCOUNT, \
     CONNECTION_INTERFACE_ALIASING, CONNECTION_INTERFACE_SIMPLE_PRESENCE

class SelectMRDialog(gtk.Dialog):

    def __init__(self, parent, coherence):
        super(SelectMRDialog, self).__init__(parent=parent,
                                             buttons = (gtk.STOCK_APPLY,
                                                        gtk.RESPONSE_ACCEPT))
        self.set_title(_("Select a MediaRenderer"))
        self.mrs = []
        self.mediarenderer_picker = hildon.PickerButton(gtk.HILDON_SIZE_FINGER_HEIGHT,
                                                        hildon.BUTTON_ARRANGEMENT_VERTICAL)
        selector = hildon.TouchSelectorEntry(text = True)
        self.mediarenderer_picker.set_title(_('Mediarenderer:'))
        for device in coherence.devices:
            if device.get_friendly_device_type().lower() == "mediarenderer":
                self.mrs.append(device)
                selector.append_text(device.get_friendly_name())

        self.mediarenderer_picker.set_selector(selector)

        if self.mrs:
            self.mediarenderer_picker.set_active(0)

        self.vbox.add(self.mediarenderer_picker)
        self.vbox.show_all()

    def get_media_renderer(self):
        selector = self.mediarenderer_picker.get_selector()
        return self.mrs[selector.get_active(0)]



class SettingsDialog(gtk.Dialog):

    def __init__(self, parent, mirabeau_section, media_server_enabled):
        super(SettingsDialog, self).__init__(parent=parent,
                                             buttons = (gtk.STOCK_SAVE,
                                                        gtk.RESPONSE_ACCEPT))
        self.set_title(_("Settings"))

        self.accounts = []
        bus = dbus.SessionBus()

        # account
        conf_account = mirabeau_section.get("account")
        index = -1
        accounts = connect.gabble_accounts()
        self.account_picker = hildon.PickerButton(gtk.HILDON_SIZE_FINGER_HEIGHT,
                                                  hildon.BUTTON_ARRANGEMENT_VERTICAL)
        selector = hildon.TouchSelectorEntry(text = True)
        self.account_picker.set_title(_('Account:'))
        for account_obj_path in accounts:
            account_obj = bus.get_object(ACCOUNT_MANAGER, account_obj_path)
            norm_name = account_obj.Get(ACCOUNT, 'NormalizedName')
            nick_name = account_obj.Get(ACCOUNT, 'Nickname')
            parameters = account_obj.Get(ACCOUNT, 'Parameters')
            if "" in (nick_name, norm_name):
                label = parameters["account"]
            else:
                label = "%s - %s" % (nick_name, norm_name)
            selector.append_text(label)
            self.accounts.append((account_obj_path, nick_name))
            if account_obj_path == conf_account:
                index = accounts.index(account_obj_path)

        self.account_picker.set_selector(selector)
        if index > -1:
            self.account_picker.set_active(index)

        self.vbox.pack_start(self.account_picker, expand=False)

        # conf server
        self.conf_server_label = gtk.Label(_("Conference Server"))
        self.conf_server_entry = hildon.Entry(gtk.HILDON_SIZE_FINGER_HEIGHT)
        self.conf_server_entry.set_text(mirabeau_section.get("conference-server", ""))
        self.conf_server_hbox = gtk.HBox()
        self.conf_server_hbox.pack_start(self.conf_server_label, expand=False)
        self.conf_server_hbox.pack_start(self.conf_server_entry, expand=True)
        self.vbox.pack_start(self.conf_server_hbox, expand=False)

        # chatroom name
        self.chatroom_label = gtk.Label(_("Chatroom"))
        self.chatroom_entry = hildon.Entry(gtk.HILDON_SIZE_FINGER_HEIGHT)
        self.chatroom_entry.set_text(mirabeau_section.get("chatroom", ""))
        self.chatroom_hbox = gtk.HBox()
        self.chatroom_hbox.pack_start(self.chatroom_label, expand=False)
        self.chatroom_hbox.pack_start(self.chatroom_entry, expand=True)
        self.vbox.pack_start(self.chatroom_hbox, expand=False)

        # MS toggle
        self.ms_toggle = hildon.CheckButton(gtk.HILDON_SIZE_FINGER_HEIGHT)
        self.ms_toggle.set_label(_("Share the media files of this device"))
        self.ms_toggle.set_active(media_server_enabled)
        self.vbox.pack_start(self.ms_toggle, expand=False)

        self.vbox.show_all()

    def get_chatroom(self):
        return self.chatroom_entry.get_text()

    def get_conf_server(self):
        return self.conf_server_entry.get_text()

    def get_account(self):
        selector = self.account_picker.get_selector()
        active = selector.get_active(0)
        path = ""
        if active != -1:
            account = self.accounts[active]
            path = account[0]
        return path

    def get_account_nickname(self):
        selector = self.account_picker.get_selector()
        active = selector.get_active(0)
        name = None
        if active != -1:
            account = self.accounts[active]
            name = account[1]
        return name

    def ms_enabled(self):
        return self.ms_toggle.get_active()

class ContactsDialog(hildon.PickerDialog):

    def __init__(self, parent, coherence):
        super(ContactsDialog, self).__init__(parent=parent)
        self.set_title(_("Select contacts"))
        self.set_done_label(_("Done"))

        roster = coherence.mirabeau.tube_publisher.roster["subscribe"]
        self.contacts = []
        selector = hildon.TouchSelector()
        store = gtk.ListStore(str)
        for handle_id, infos in roster.iteritems():
            alias = infos["%s/alias" % CONNECTION_INTERFACE_ALIASING]
            presence = infos["%s/presence" % CONNECTION_INTERFACE_SIMPLE_PRESENCE]
            label = "%s (%s)" % (alias, presence[1])
            item_iter = store.append()
            store.set(item_iter, 0, label)
            self.contacts.append((handle_id, alias))

        column = selector.append_text_column(store, 0)

        renderer = gtk.CellRendererText()
        column.pack_start(renderer)

        selector.set_column_selection_mode(hildon.TOUCH_SELECTOR_SELECTION_MODE_MULTIPLE)
        self.set_selector(selector)

    def get_contacts(self):
        selector = self.get_selector()
        indexes = [ c[0] for c in selector.get_selected_rows(0)]
        contacts = [self.contacts[idx] for idx in indexes]
        return contacts
