#
# Copyright 2023 Thomas Engel <realdatenwurm@gmail.com>
# License:  same as zim (gpl)
#
# DESCRIPTION:
#
# Zim plugin providing a convenient way to generate text using ChatGPT.
#
# CHANGELOG:
#
# 2023-05-06 Initial version
# 2023-05-07 Refactored code

import logging
import threading

try:
    import openai
except ImportError:
    openai = None

try:
    import gi

    gi.require_version('Gdk', '3.0')
    gi.require_version('Gtk', '3.0')
    from gi.repository import Gdk, Gtk, GObject, GLib
except ImportError:
    gi = None

from zim.actions import action
from zim.config import ConfigDefinition
from zim.gui.mainwindow import MainWindowExtension
from zim.gui.widgets import Dialog, ErrorDialog
from zim.plugins import PluginClass

logger = logging.getLogger('zim.plugins.gpt')

DEFAULT_MODEL = 'gpt-3.5-turbo'
DEFAULT_TEMPERATURE = "0.7"
DEFAULT_MAX_TOKENS = 1200
DEFAULT_REPLACE_SELECTED_TEXT = True


class TemperatureInRangeCheck(ConfigDefinition):

    def check(self, value):
        if not 0 <= float(value) <= 1.0:
            raise ValueError('Value needs to be within range of 0.0 and 1.0')
        return value


class ZimGPTPlugin(PluginClass):
    plugin_info = {
        'name': _('ZimGPT'),  # T: plugin name
        'description': _(
            'This plugin provides a convenient way to generate text using ChatGPT.'
            'With this plugin, you can select a piece of text and instruct ChatGPT '
            'to generate additional text based on the selected content.\n\n'
            'The generated text can be used for a variety of purposes, such as completing '
            'a thought, generating new ideas, or expanding on a concept.\n\n'
            'To use the plugin, simply select the desired text and press the keyboard shortcut '
            'Alt+Shift+I to open a dialog box to enter your instruction. The keyboard shortcut '
            'can be customized to suit your preferences using Zim\'s key bindings preferences.'
        ),  # T: plugin description
        'author': 'Thomas Engel <realdatenwurm@gmail.com>',
        'help': 'Plugins:ZimGPT',
    }

    plugin_preferences = (
        # key, type, label, default
        ('api_key', 'string', _('OpenAI API Key'), 'Please enter OPENAI API Key'),
        ('model', 'string', _('Model'), DEFAULT_MODEL),
        ('temperature', 'string', _('Temperature'), DEFAULT_TEMPERATURE, TemperatureInRangeCheck),
        ('max_tokens', 'int', _('Max Tokens'), DEFAULT_MAX_TOKENS, (0, 4096)),
        ('replace_selected_text', 'bool', None, DEFAULT_REPLACE_SELECTED_TEXT),
    )

    @classmethod
    def check_dependencies(self):
        return bool(openai or gi), [
            ('openai', not openai is None, True),
            ('gi', not gi is None, True),
        ]


class ZimGPTMainWindowExtension(MainWindowExtension):
    """ Listener for the ZimGPT plugin. """

    def __init__(self, plugin, window):
        MainWindowExtension.__init__(self, plugin, window)
        self.preferences = plugin.preferences
        self.window = window

    @action('', accelerator='<alt><shift>i', menuhints='accelonly')
    def do_instruct(self):
        if not self._validate_preferences():
            return

        text_buffer = TextBuffer(self.window.pageview.textview.get_buffer())
        dialog = InstructionDialog(self.window, _("Enter Instruction"),
                                   show_replace_selected_text_choice=text_buffer.has_selection())
        dialog.set_replace_selected_text_choice(self.replace_selected_text_choice)
        dialog_response = dialog.run()

        # Remember user choice
        self.replace_selected_text_choice = dialog.get_replace_selected_text_choice()

        if dialog_response != Gtk.ResponseType.OK:
            dialog.destroy()
            return

        if text_buffer.has_selection():
            selected_text = text_buffer.get_selected_text()
            prompt = f"Please apply the following instruction/question for the supplied text:\n\n" \
                     f"Instruction: \"\"\"\n" \
                     f"{dialog.get_instruction()}\n" \
                     f"\"\"\"\n\n" \
                     f"Text: \"\"\"\n" \
                     f"{selected_text}\n" \
                     f"\"\"\"\n\n" \
                     f"Output:"
        else:
            prompt = f"Please generate a text following the instructions below:\n\n" \
                     f"Instruction: \"\"\"\n" \
                     f"{dialog.get_instruction()}\n" \
                     f"\"\"\"\n\n" \
                     f"Output:"

        task = GPTTask(self.preferences, title=_('Generate Text'), prompt=prompt)
        self._execute_task(task)
        if not task.result:
            # BUG: Text selection in page view may get lost when opening the dialog.
            # FIX: Cache text selection and restore after dialog is closed.
            text_buffer.restore_selection_bounds()
            return

        text_buffer.insert_at_cursor(task.result, dialog.get_replace_selected_text_choice())

    def _validate_preferences(self):
        try:
            openai.api_key = self.api_key
            models = [model.id for model in openai.Model.list().data]
            if self.model not in models:
                ErrorDialog(self, (
                    _('Error'),
                    # T: error message
                    _('The selected model does not exist'),
                    # T: error message explanation
                )).run()
            return True
        except openai.error.AuthenticationError as err:
            logger.exception(err)
            ErrorDialog(self, (
                _('Error'),
                # T: error message
                _('There was an error with the authentication credentials'),
                # T: error message explanation
            )).run()
        except openai.error.APIConnectionError as err:
            logger.exception(err)
            ErrorDialog(self, (
                _('Error'),
                # T: error message
                _('There was an error connecting to the OpenAI API'),
                # T: error message explanation
            )).run()
        return False

    def _execute_task(self, task):
        """ Executes a given task while showing a pulsed progress dialog. """
        try:
            progress_dialog = PulsedProgressDialog(self.window, task)
            progress_dialog.run()
            progress_dialog.destroy()
        except Exception as err:
            logger.exception(err)
            ErrorDialog(self, (
                _('Error'),
                # T: error message
                _('There was an unknown error while processing the task'),
                # T: error message explanation
            )).run()

    @property
    def api_key(self):
        return self.preferences['api_key']

    @property
    def model(self):
        return self.preferences['model']

    @property
    def replace_selected_text_choice(self):
        return self.preferences['replace_selected_text']

    @replace_selected_text_choice.setter
    def replace_selected_text_choice(self, value):
        self.preferences['replace_selected_text'] = value


class TextBuffer:
    """ Helper class for a textview buffer. """

    def __init__(self, buffer):
        self._buffer = buffer
        self._save_selection_bounds()

    def has_selection(self):
        return self._sel_start is not None and self._sel_end is not None

    def _save_selection_bounds(self):
        if self._buffer.get_has_selection():
            self._sel_start, self._sel_end = self._buffer.get_selection_bounds()
        else:
            self._sel_start, self._sel_end = None, None

    def get_selected_text(self):
        if self.has_selection():
            return self._buffer.get_text(self._sel_start, self._sel_end, False)

    def insert_at_cursor(self, new_text, do_replace_selection=True):
        if self.has_selection() and do_replace_selection:
            self._buffer.delete(self._sel_start, self._sel_end)
        self._buffer.insert_at_cursor(new_text)

    def restore_selection_bounds(self):
        if self.has_selection():
            self._buffer.select_range(self._sel_start, self._sel_end)


class GPTTask:

    def __init__(self, preferences, title, prompt):
        self.title = title
        self.preferences = preferences
        self.prompt = prompt
        self.result = ""

    def execute(self):
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "user", "content": self.prompt}
            ],
            max_tokens=self.max_tokens,
            n=1,
            stop=None,
            temperature=float(self.temperature),
        )
        self.result = response['choices'][0]['message']['content'].strip()

    @property
    def api_key(self):
        return self.preferences['api_key']

    @property
    def model(self):
        return self.preferences['model']

    @property
    def max_tokens(self):
        return self.preferences['max_tokens']

    @property
    def temperature(self):
        return self.preferences['temperature']


class PulsedProgressDialog(Gtk.Dialog):

    def __init__(self, parent, task):
        Gtk.Dialog.__init__(self, title=task.title, transient_for=parent, modal=True)
        self.task = task
        self.progress_bar = Gtk.ProgressBar()
        self.vbox.pack_start(self.progress_bar, False, False, 0)
        self.vbox.show_all()
        self.set_size_request(400, 20)
        self._start_progress()

    def _start_progress(self):
        self.progress_bar.show()

        # Set up the progress bar pulse
        self.progress_bar_pulse_source = GLib.timeout_add(100, self.update_progress_bar_pulse)

        # Start the summary generation thread
        thread = threading.Thread(target=self._thread)
        thread.daemon = True
        thread.start()

    def update_progress_bar_pulse(self):
        self.progress_bar.pulse()
        return True  # Continue pulsing

    def finish_progress(self):
        # Stop pulsing the progress bar
        GLib.source_remove(self.progress_bar_pulse_source)

        # Update progress bar to 100%
        self.progress_bar.set_fraction(1)

        # Close the dialog
        self.response(Gtk.ResponseType.OK)

    def _thread(self):
        self.task.execute()
        GObject.idle_add(self.finish_progress)


class InstructionDialog(Dialog):
    def __init__(self, parent, title, show_replace_selected_text_choice):
        Dialog.__init__(self, parent, title)

        self._instruction = None
        self._replace_selection = True

        # Create the instruction label and entry
        self._entry = Gtk.Entry()
        self._entry.set_activates_default(True)  # Make ENTER key press trigger the OK button.
        self._entry.set_placeholder_text(_("Enter an instruction here..."))
        self._entry.connect("changed", self.on_entry_changed)

        self._chk_replace_selected_text = Gtk.CheckButton()
        self._chk_replace_selected_text.set_label(_("Replace selected text"))
        self._chk_replace_selected_text.set_active(True)
        self._chk_replace_selected_text.connect("toggled", self.on_replace_selection_toggled)

        # Add ok button.
        self.btn_ok = self.get_widget_for_response(response_id=Gtk.ResponseType.OK)
        self.btn_ok.set_can_default(True)
        self.btn_ok.grab_default()

        self.hbox = Gtk.HBox()
        self.hbox.pack_start(self._entry, True, True, 0)
        self.vbox.pack_start(self.hbox, True, True, 0)

        if show_replace_selected_text_choice:
            # Only show checkbox when a text is selected
            self.vbox.pack_start(self._chk_replace_selected_text, True, True, 0)

        # Configure dialog.
        self.set_modal(True)
        self.set_default_size(380, 100)

        # Set focus to search field
        self._entry.grab_focus()

        # Disable the OK button initially
        self.set_response_sensitive(Gtk.ResponseType.OK, False)

    def on_replace_selection_toggled(self, widget):
        self._replace_selection = self._chk_replace_selected_text.get_active()

    def on_entry_changed(self, widget):
        # Enable the OK button if the entry is not empty
        text = self._entry.get_text().strip()
        self.set_response_sensitive(Gtk.ResponseType.OK, bool(text))

    def get_instruction(self):
        return self._instruction

    def set_replace_selected_text_choice(self, choice):
        self._replace_selection = choice
        self._chk_replace_selected_text.set_active(choice)

    def get_replace_selected_text_choice(self):
        return self._replace_selection

    def do_response_ok(self):
        self.result = Gtk.ResponseType.OK
        self._instruction = self._entry.get_text()
        return True
