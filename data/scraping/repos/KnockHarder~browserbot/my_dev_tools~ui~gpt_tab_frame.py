import asyncio
import datetime
import os.path
import sys
from typing import Optional, Coroutine

from PySide6.QtCore import Slot, Signal, Qt
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import QFrame, QWidget, QFileDialog, QPlainTextEdit, QApplication, QMessageBox, \
    QInputDialog, QTableWidgetItem
from jinja2 import TemplateError
from langchain.prompts import load_prompt, PromptTemplate

from ..config import gpt_prompt_file_dir
from ..gpt import ChatGptPage
from ..gpt import parse_template


def safe_parse_template(parent: QWidget, template: str) -> Optional[PromptTemplate]:
    try:
        return parse_template(template)
    except (ValueError, TemplateError) as e:
        message_box = QMessageBox(QMessageBox.Icon.Critical, '解析模板失败',
                                  '内容不合法', QMessageBox.StandardButton.Ok, parent)
        message_box.setDetailedText(str(e))
        message_box.open()
        return None


class GptTabFrame(QFrame):
    ROLE_IS_STATIC_ROW = Qt.ItemDataRole.UserRole + 1
    templateTextReset = Signal(str)
    statusLabelTextReset = Signal(str)
    answerTextReset = Signal(str)
    template_edit_widget: QPlainTextEdit

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.chat_page = ChatGptPage()

        from ..ui.gpt_tab_frame_uic import Ui_GptTabFrame
        self.ui = Ui_GptTabFrame()
        self.ui.setupUi(self)
        self.template_file = None
        self.init_prompt_inputs()
        self.task_info_list = list()

        QShortcut('Ctrl+Shift+Backspace', self, self.clear_chat_history)
        QShortcut(QKeySequence.StandardKey.AddTab, self, self.new_chat)
        QShortcut("Ctrl+Return", self, self.generate_code)
        QShortcut(QKeySequence.StandardKey.Open, self, self.load_template_for_chat)
        QShortcut(QKeySequence.StandardKey.Cancel, self, self.cancel_future)
        QShortcut(QKeySequence.StandardKey.Save, self, self.save_template)
        QShortcut(QKeySequence.StandardKey.Refresh, self, self.rename_template_file)

    def init_prompt_inputs(self):
        table = self.ui.prompt_input_table
        table.setColumnCount(1)
        table.verticalHeader().setLineWidth(0)

        table.insertRow(0)
        item = QTableWidgetItem('模板')
        item.setData(self.ROLE_IS_STATIC_ROW, True)
        table.setVerticalHeaderItem(0, item)
        self.template_edit_widget = QPlainTextEdit(table)
        self.templateTextReset.connect(self.template_edit_widget.setPlainText)
        self.template_edit_widget.textChanged.connect(self.update_for_template_change)
        table.setCellWidget(0, 0, self.template_edit_widget)

    def cancel_future(self):
        while self.task_info_list:
            task_info = self.task_info_list.pop()
            task = task_info['task']
            task.cancel()
            self.statusLabelTextReset.emit(f'{task.get_name()}被中断')
            task_info['cancelCallback']()

    @Slot()
    def new_chat(self):
        async def _do_async():
            await self.chat_page.new_chat()
            await self.chat_page.activate()

        self._create_task(_do_async(), 'New Chat')

    @staticmethod
    def _create_task(coro: Coroutine, name: str):
        asyncio.create_task(coro, name=name)

    @Slot()
    def clear_chat_history(self):
        asyncio.create_task(self.chat_page.clear_histories(), name='clear history')

    @Slot()
    def load_template_for_chat(self):
        def _load_file(path):
            prompt = load_prompt(path)
            self.template_file = path
            self.templateTextReset.emit(prompt.template)
            self.update_variable_form(prompt.template)
            self.statusLabelTextReset.emit(f'加载文件: {path}')

        dialog = QFileDialog(self, '打开模板文件', gpt_prompt_file_dir(), 'JSON Files(*.json)')
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.fileSelected.connect(_load_file)
        dialog.open()

    @Slot()
    def update_for_template_change(self):
        template = self.template_edit_widget.toPlainText()
        self.statusLabelTextReset.emit('模板被修改')
        self.update_variable_form(template)

    def update_variable_form(self, template: str):
        try:
            prompt = parse_template(template)
        except Exception:
            self.statusLabelTextReset.emit('模板不合法')
            return
        table = self.ui.prompt_input_table
        start_row = self.static_row_count(table)
        variables = sorted(prompt.input_variables, key=lambda x: template.index(x))
        for idx, variable in enumerate(variables):
            row = next(filter(lambda x: table.verticalHeaderItem(x).text() == variable,
                              range(table.rowCount())),
                       None)
            if not row:
                row = start_row + idx
                table.insertRow(row)
                table.setVerticalHeaderItem(row, QTableWidgetItem(variable))
                edit_widget = QPlainTextEdit(table)
                table.setCellWidget(row, 0, edit_widget)
            elif row < start_row:
                raise ValueError(f'变量{variable}与静态变量重复')
        row = start_row
        while row < table.rowCount():
            variable = table.verticalHeaderItem(row).text()
            try:
                idx = variables.index(variable)
            except ValueError:
                table.cellWidget(row, 0).deleteLater()
                table.removeRow(row)
                continue
            if start_row + idx != row:
                header = table.verticalHeader()
                header.moveSection(header.logicalIndex(row), header.logicalIndex(idx + start_row))
            else:
                row += 1

    def static_row_count(self, table):
        return len([i for i in range(table.rowCount())
                    if table.verticalHeaderItem(i).data(self.ROLE_IS_STATIC_ROW) is True])

    @Slot()
    def generate_code(self):
        submit_btn = self.ui.submitBtn
        if not submit_btn.isEnabled():
            return
        template = self.template_edit_widget.toPlainText()
        if not template:
            return
        try:
            prompt = parse_template(template)
        except Exception:
            self.statusLabelTextReset.emit(f'模板不合法，请修正后再提交{template}')
            return
        param_map = dict()
        table = self.ui.prompt_input_table
        row_start = self.static_row_count(table)
        for i in range(row_start, table.rowCount()):
            variable = table.verticalHeaderItem(i).text()
            input_widget = table.cellWidget(i, 0)
            param_map[variable] = input_widget.toPlainText()

        async def async_gen_code():
            submit_btn.setEnabled(False)
            self.statusLabelTextReset.emit('正在生成代码...')
            answer = await self.chat_page.gen_code_question(prompt, **param_map)
            self.answerTextReset.emit(answer)
            self.statusLabelTextReset.emit('生成成功!')
            self.activate_window()
            submit_btn.setEnabled(True)

        self.task_info_list.append({
            'task': asyncio.create_task(
                async_gen_code(), name=f'{datetime.datetime.now().strftime("%H:%M:%S")} 提交的代码生成任务'),
            'cancelCallback': lambda: submit_btn.setEnabled(True)
        })

    def activate_window(self):
        parent = self.parent()
        while parent.parent():
            parent = parent.parent()
        parent.activateWindow()
        parent.raise_()

    @Slot()
    def save_template(self):
        def _save_template(path: str):
            template = self.template_edit_widget.toPlainText()
            prompt = safe_parse_template(self, template)
            if not prompt:
                return
            prompt.save(path)
            self.template_file = path
            self.ui.rename_template_btn.setEnabled(True)
            self.statusLabelTextReset.emit(f'模板保存至: {path}')

        default_path = self.template_file if self.template_file else gpt_prompt_file_dir()
        dialog = QFileDialog(self, '保存至', default_path, 'JSON Files(*.json')
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.fileSelected.connect(_save_template)
        dialog.open()

    @Slot()
    def rename_template_file(self):
        file_path = self.template_file
        if not file_path:
            self.statusLabelTextReset.emit('请先加载模板文件')
            return
        if not os.path.exists(file_path):
            self.statusLabelTextReset.emit(f'文件已被删除: {file_path}')
            return
        self.load_template_for_chat(file_path)
        while True:
            new_name, confirmed = QInputDialog.getText(self, '模板文件重命名', '新的文件名')
            if not confirmed:
                return
            new_path = os.path.join(os.path.dirname(file_path), new_name)
            if not new_path.endswith('.json'):
                new_path += '.json'
            if os.path.exists(new_path):
                dialog = QMessageBox(QMessageBox.Icon.Critical, '重命名失败', f'{new_path}文件已存在')
                dialog.open()
            else:
                os.rename(file_path, new_path)
                self.template_file = new_path
                self.statusLabelTextReset.emit(f'文件更名为: {new_path}')
                return


if __name__ == '__main__':
    def main():
        app = QApplication()
        frame = GptTabFrame()
        frame.show()
        sys.exit(app.exec())


    main()
