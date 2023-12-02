# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Implementation of tool support over LSP."""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
import sys
from typing import Any, Literal, Optional


# **********************************************************
# Update sys.path before importing any bundled libraries.
# **********************************************************
def update_sys_path(path_to_add: str, strategy: str) -> None:
    """Add given path to `sys.path`."""
    if path_to_add not in sys.path and os.path.isdir(path_to_add):
        if strategy == "useBundled":
            sys.path.insert(0, path_to_add)
        elif strategy == "fromEnvironment":
            sys.path.append(path_to_add)


# Ensure that we can import LSP libraries, and other bundled libraries.
update_sys_path(
    os.fspath(pathlib.Path(__file__).parent.parent / "libs"),
    "useBundled"
)


# **********************************************************
# Imports needed for the language server goes below this.
# **********************************************************
# pylint: disable=wrong-import-position,import-error
import lsprotocol.types as lsp
from pygls import server, uris, workspace

import lsp_jsonrpc as jsonrpc
from lsp_custom_types import TelemetryParams, TelemetryTypes
from lsp_progress import Progress, ProgressHundlers


class LanguageServer(server.LanguageServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lsp.progress_handlers = ProgressHundlers()
        self.lsp.fm.add_builtin_feature(lsp.WINDOW_WORK_DONE_PROGRESS_CANCEL, self.progress_cancel)

    def progress(self, *args, **kwargs):
        return Progress(self.lsp, *args, **kwargs)

    def progress_cancel(ls: server.LanguageServer,
                        params: lsp.WorkDoneProgressCancelParams):
        ls.lsp.progress_handlers.get(params.token).cancel()

    def apply_edit_async(
        self, edit: lsp.WorkspaceEdit, label: Optional[str] = None
    ) -> lsp.WorkspaceApplyEditResponse:
        """Sends apply edit request to the client. Should be called with `await`"""
        return self.lsp.send_request_async(
            lsp.WORKSPACE_APPLY_EDIT,
            lsp.ApplyWorkspaceEditParams(edit=edit, label=label)
        )

    def _send_telemetry(self, params: TelemetryParams):
        self.send_notification(lsp.TELEMETRY_EVENT, params)

    def send_telemetry_info(self, name: str, data: dict[str, str]):
        params = TelemetryParams(TelemetryTypes.Info, name, data)
        self._send_telemetry(params)

    def send_telemetry_error(self, name: str, data: dict[str, str]):
        params = TelemetryParams(TelemetryTypes.Error, name, data)
        self._send_telemetry(params)


WORKSPACE_SETTINGS = {}
GLOBAL_SETTINGS = {}
MAX_WORKERS = 5
LSP_SERVER = LanguageServer(
    name="chatgpt-docstrings", version="0.1", max_workers=MAX_WORKERS
)
TOOL_MODULE = "chatgpt-docstrings"
TOOL_DISPLAY = "ChatGPT: Docstring Generator"


# **********************************************************
# Required Language Server Initialization and Exit handlers.
# **********************************************************
@LSP_SERVER.feature(lsp.INITIALIZE)
def initialize(params: lsp.InitializeParams) -> None:
    """LSP handler for initialize request."""
    log_to_output(f"CWD Server: {os.getcwd()}")
    log_to_output(f"PID Server: {os.getpid()}")

    paths = "\r\n   ".join(sys.path)
    log_to_output(f"sys.path used to run Server:\r\n   {paths}")

    GLOBAL_SETTINGS.update(
        **params.initialization_options.get("globalSettings", {})
    )

    settings = params.initialization_options["settings"]
    _update_workspace_settings(settings)
    settings_output = json.dumps(settings,
                                 indent=4,
                                 ensure_ascii=False)
    global_settings_output = json.dumps(GLOBAL_SETTINGS,
                                        indent=4,
                                        ensure_ascii=False)
    log_to_output(f"Settings used to run Server:\r\n{settings_output}\r\n")
    log_to_output(f"Global settings:\r\n{global_settings_output}\r\n")


@LSP_SERVER.feature(lsp.EXIT)
def on_exit(_params: Optional[Any] = None) -> None:
    """Handle clean up on exit."""
    jsonrpc.shutdown_json_rpc()


@LSP_SERVER.feature(lsp.SHUTDOWN)
def on_shutdown(_params: Optional[Any] = None) -> None:
    """Handle clean up on shutdown."""
    jsonrpc.shutdown_json_rpc()


def _get_global_defaults():
    return {
        **GLOBAL_SETTINGS,
        "interpreter": GLOBAL_SETTINGS.get("interpreter", [sys.executable]),
    }


def _update_workspace_settings(settings):
    if not settings:
        key = os.getcwd()
        WORKSPACE_SETTINGS[key] = {
            "cwd": key,
            "workspaceFS": key,
            "workspace": uris.from_fs_path(key),
            **_get_global_defaults(),
        }
        return

    for setting in settings:
        key = uris.to_fs_path(setting["workspace"])
        WORKSPACE_SETTINGS[key] = {
            **setting,
            "workspaceFS": key,
        }


def _get_settings_by_path(file_path: pathlib.Path):
    workspaces = {s["workspaceFS"] for s in WORKSPACE_SETTINGS.values()}

    while file_path != file_path.parent:
        str_file_path = str(file_path)
        if str_file_path in workspaces:
            return WORKSPACE_SETTINGS[str_file_path]
        file_path = file_path.parent

    setting_values = list(WORKSPACE_SETTINGS.values())
    return setting_values[0]


def _get_document_key(document: workspace.Document):
    if WORKSPACE_SETTINGS:
        document_workspace = pathlib.Path(document.path)
        workspaces = {s["workspaceFS"] for s in WORKSPACE_SETTINGS.values()}

        # Find workspace settings for the given file.
        while document_workspace != document_workspace.parent:
            if str(document_workspace) in workspaces:
                return str(document_workspace)
            document_workspace = document_workspace.parent

    return None


def _get_settings_by_document(document: workspace.Document | None):
    if document is None or document.path is None:
        return list(WORKSPACE_SETTINGS.values())[0]

    key = _get_document_key(document)
    if key is None:
        # This is either a non-workspace file or there is no workspace.
        key = os.fspath(pathlib.Path(document.path).parent)
        return {
            "cwd": key,
            "workspaceFS": key,
            "workspace": uris.from_fs_path(key),
            **_get_global_defaults(),
        }

    return WORKSPACE_SETTINGS[str(key)]


# **********************************************************
# Generate docstring features start here
# **********************************************************
import openai

from code_parser import FuncParser, NotFuncException, Position, Range


@LSP_SERVER.command("chatgpt-docstrings.applyGenerate")
async def apply_generate_docstring(ls: server.LanguageServer,
                                   args: list[lsp.TextDocumentPositionParams, str]):
    uri = args[0]["textDocument"]["uri"]
    openai_api_key = args[1]
    progress_token = args[2]
    document = ls.workspace.get_document(uri)
    source = document.source
    cursor = args[0]["position"]
    cursor["line"] += 1
    cursor = Position(*cursor.values())
    settings = _get_settings_by_document(document)
    openai_model = settings["openaiModel"]
    prompt_pattern = settings["promptPattern"]
    docstring_format = settings["docstringFormat"]
    response_timeout = settings["responseTimeout"]

    # get function source
    try:
        func = FuncParser(source, cursor)
    except NotFuncException:
        show_info("The cursor must be set inside the function.")
        return

    # format prompt
    prompt = prompt_pattern.format(docstring_format=docstring_format,
                                   function=func.code)
    log_to_output(f"Used ChatGPT prompt:\n{prompt}")

    # get gocstring
    with ls.progress(progress_token) as progress:
        task = asyncio.create_task(_get_docstring(openai_api_key, openai_model, prompt))
        while 1:
            if task.done():
                break
            if response_timeout == 0:
                task.cancel()
                show_warning("ChatGPT response timed out.")
                return
            if progress.cancelled:
                task.cancel()
                return
            progress.report(f"Waiting for ChatGPT response ({response_timeout} secs)...")
            await asyncio.sleep(1)
            response_timeout -= 1
        if task.exception():
            raise task.exception()
        docstring = task.result()
    log_to_output(f"Received ChatGPT docstring:\n{docstring}")

    # format docstring
    docstring = _format_docstring(docstring, func.indent_level+1)
    docstring = _match_line_endings(document, docstring)

    # define docsting position
    if func.docstring_range:
        docstring_pos = Range(func.suite, func.docstring_range.end)
    else:
        docstring_pos = Range(func.suite, func.suite)

    # apply docstring
    text_edits = _create_text_edits(docstring_pos, docstring)
    workspace_edit = _create_workspace_edit(document, text_edits)
    result = await ls.apply_edit_async(workspace_edit)
    if not result.applied:
        reason = result.failure_reason or \
            "maybe you make changes to source code at generation time"
        show_warning(f"Failed to add docstring to source code ({reason})")
        ls.send_telemetry_error('applyEditWorkspaceFail', {'reason': reason})


async def _get_docstring(api_key: str,
                   model: Literal["gpt-3.5-turbo", "text-davinci-002"],
                   prompt: str) -> str:
    openai.api_key = api_key
    if model == "gpt-3.5-turbo":
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {"role": "system",
                 "content": "When you generate a docstring, return me only a string that I can add to my code."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        docstring = response.choices[0].message.content
    elif model == "text-davinci-002":
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=0,
            max_tokens=1000,
        )
        docstring = response["choices"][0]["text"]
    else:
        raise Exception(
            'Only models "gpt-3.5-turbo" and "text-davinci-002" are supported!'
        )
    return docstring


def _format_docstring(docstring: str, indent_level: int) -> str:
    # remove function source code including markdown tags
    if docstring.strip().startswith(("def ", "async ", "```")):
        match = re.search(r'""".*?"""', docstring, flags=re.DOTALL)
        docstring = match.group() if match else docstring
    # remove leading and trailing whitespaces, newlines, quotes
    docstring = docstring.strip().strip('"""').strip("\r\n")
    # remove indents
    if docstring.startswith("    "):
        lines = docstring.splitlines(True)
        docstring = "".join([re.sub(r"^\s{4}", "", line) for line in lines])
    # eol conversion to single format
    docstring = "\n".join(docstring.splitlines())
    # add quotes
    docstring = f'"""{docstring}\n"""'
    # add indents
    indents = " "*indent_level*4
    docstring = "".join([f"{indents}{line}" for line in docstring.splitlines(True)])
    # add new line
    docstring = f"\n{docstring}"
    return docstring


def _create_text_edits(docstring_pos: Range, docstring: str) -> list[lsp.TextEdit]:
    return [
        lsp.TextEdit(
            range=Range(
                start=Position(
                    line=docstring_pos.start.line - 1,
                    character=docstring_pos.start.character,
                ),
                end=Position(
                    line=docstring_pos.end.line - 1,
                    character=docstring_pos.end.character,
                ),
            ),
            new_text=docstring,
        )
    ]


def _create_workspace_edit(
    document: lsp.Document, text_edits: list[lsp.TextEdit]
) -> lsp.WorkspaceEdit:
    return lsp.WorkspaceEdit(
        document_changes=[
            lsp.TextDocumentEdit(
                text_document=lsp.VersionedTextDocumentIdentifier(
                    uri=document.uri,
                    version=0 if document.version is None else document.version,
                ),
                edits=text_edits,
            )
        ]
    )


def _get_line_endings(lines: list[str]) -> str:
    """Returns line endings used in the text."""
    try:
        if lines[0][-2:] == "\r\n":
            return "\r\n"
        return "\n"
    except Exception:  # pylint: disable=broad-except
        return None


def _match_line_endings(document: workspace.Document, text: str) -> str:
    """Ensures that the edited text line endings matches the document line endings."""
    expected = _get_line_endings(document.source.splitlines(keepends=True))
    actual = _get_line_endings(text.splitlines(keepends=True))
    if actual == expected or actual is None or expected is None:
        return text
    return text.replace(actual, expected)
# **********************************************************
# Generate docstring features ends here
# **********************************************************


# *****************************************************
# Logging and notification.
# *****************************************************
def log_to_output(message: str,
                  msg_type: lsp.MessageType = lsp.MessageType.Log) -> None:
    LSP_SERVER.show_message_log(message, msg_type)


def show_error(message: str) -> None:
    log_to_output(message, lsp.MessageType.Error)
    LSP_SERVER.show_message(message, lsp.MessageType.Error)


def show_warning(message: str) -> None:
    log_to_output(message, lsp.MessageType.Warning)
    LSP_SERVER.show_message(message, lsp.MessageType.Warning)


def show_info(message: str) -> None:
    log_to_output(message, lsp.MessageType.Info)
    LSP_SERVER.show_message(message, lsp.MessageType.Info)


# *****************************************************
# Start the server.
# *****************************************************
if __name__ == "__main__":
    LSP_SERVER.start_io()
