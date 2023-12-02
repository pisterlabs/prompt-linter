from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog, QCheckBox
from qtpy.QtCore import QTimer

from pyqode.python.backend import server
from pyqode.python.widgets import PyCodeEdit

from ._scripts_directory import _init_scripts_directory, _new_template_filename, _exec_code
import os
from napari_tools_menu import register_dock_widget

@register_dock_widget(menu="Scripts > Script Editor")
class ScriptEditor(QWidget):
    def __init__(self, napari_viewer, _for_testing:bool = False):
        super().__init__()

        if 'QT_API' not in os.environ:
            os.environ['QT_API'] = 'pyqt5'

        self._viewer = napari_viewer
        ScriptEditor._add_script_editor(self)

        wgt = QWidget()
        wgt.setLayout(QHBoxLayout())

        btn = QPushButton("New")
        btn.clicked.connect(self._on_new_click)
        wgt.layout().addWidget(btn)

        btn = QPushButton("Load")
        btn.clicked.connect(self._on_load_click)
        wgt.layout().addWidget(btn)

        btn = QPushButton("Save")
        btn.clicked.connect(self._on_save_click)
        wgt.layout().addWidget(btn)

        btn = QPushButton("Run")
        btn.clicked.connect(self._on_run_click)
        wgt.layout().addWidget(btn)

        try:
            import openai
            btn = QPushButton("Ask chatGPT")
            btn.clicked.connect(self._on_ask_chat_gpt_click)
            wgt.layout().addWidget(btn)
        except:
            pass

        recorder_avaliable = False
        try:
            import napari_workflows
            recorder_avaliable = True
        except:
            pass

        if recorder_avaliable:
            chb_record = QCheckBox("Record")
            wgt.layout().addWidget(chb_record)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(wgt)

        self._code_edit = PyCodeEdit(server_script=server.__file__)
        self.layout().addWidget(self._code_edit)

        self._code_edit.setStyleSheet("background-color: rgb(196, 196, 196);")
        self._on_new_click()

        if recorder_avaliable and not _for_testing:
            #
            self.timer = QTimer()
            self.timer.setInterval(200)

            @self.timer.timeout.connect
            def update_recorded_code(*_):
                try:
                    if not chb_record.isChecked():
                        return
                except:
                    self.timer.stop()

                from napari_workflows import WorkflowManager
                complete_code = WorkflowManager.install(self._viewer).to_python_code()
                self.set_code(complete_code)

            self.timer.start()

    def _on_new_click(self):
        new_template = _init_scripts_directory() + _new_template_filename()
        self._code_edit.file.open(new_template)

    def _on_load_click(self, event = None, filename:str = None):
        if filename is None:
            filename,_ = QFileDialog.getOpenFileName(parent=self, filter="*.py", directory=_init_scripts_directory())
        if os.path.isfile(filename):
            self._code_edit.file.open(filename)

    def _on_save_click(self, event = None, filename:str = None):
        if filename is None:
            filename, _ = QFileDialog.getSaveFileName(parent=self, filter="*.py", directory=_init_scripts_directory())

        if isinstance(filename, str) and len(filename) > 0:
            self._code_edit.file.save(filename)

    def _on_run_click(self):
        _exec_code(self._code_edit.toPlainText(), self._viewer)

    def _on_ask_chat_gpt_click(self):
        from ._chatgpt import ask_chat_gpt
        response = ask_chat_gpt(self._code_edit.textCursor().selectedText())
        self._code_edit.textCursor().insertText(response)


    def set_code(self, code):
        if self._code_edit.toPlainText() != code:
            cursor_position = self._code_edit.textCursor().position()
            self._code_edit.setPlainText(code)
            cursor = self._code_edit.textCursor()
            cursor.setPosition(cursor_position)
            self._code_edit.setTextCursor(cursor)

    @classmethod
    def _add_script_editor(cls, instance):
        if not hasattr(cls, "editors"):
            cls.editors = []
        cls.editors.append(instance)

    @classmethod
    def get_script_editor_from_viewer(cls, viewer, create_editor=True, _for_testing:bool=False):
        result = None
        if hasattr(cls, "editors"):
            # search for the last editor which was added
            for editor in cls.editors:
                if editor._viewer is viewer:
                    result = editor
        if result is None:
            if create_editor:
                result = ScriptEditor(viewer, _for_testing)
                w = viewer.window.add_dock_widget(result, area='right', name="Script editor")
                if not _for_testing:
                    w.setFloating(True)
                w.resize(600, 400)

        return result

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [ScriptEditor]
