#!/usr/bin/env python3

from typing import TYPE_CHECKING, Callable, NoReturn
import sys
from .prompts import (
    RAW_TEMPLATE,
    RAW_TRIPLE_QUOTES_TEMPLATE,
    REPLY_OK_IF_YOU_READ_TEMPLATE,
    REPLY_OK_IF_YOU_READ_TEMPLATE_SPLITTED_FIRST,
    REPLY_OK_IF_YOU_READ_TEMPLATE_SPLITTED_CONTINUED,
)


if TYPE_CHECKING:
    from datetime import datetime
    from langchain.docstore.document import Document


def convert_str_slice_notation_to_slice(str_slice: str) -> slice:
    # '1:3' -> slice(1, 3)
    # '1:' -> slice(1, None)
    # ':3' -> slice(None, 3)
    # ':' -> slice(None, None)
    # '3' -> slice(3)
    # '1:8:2' -> slice(1, 8, 2)
    # start
    def int_or_none(s: str) -> int | None:
        try:
            return int(s)
        except ValueError:
            return None

    return slice(*list(map(int_or_none, str_slice.split(':'))))
    # if str_slice.startswith(':'):
    #     start = None
    # else:
    #     start = int(str_slice.split(':')[0])
    # # stop
    # if str_slice.endswith(':'):
    #     stop = None
    # else:
    #     try:
    #         stop = int(str_slice.split(':')[1])
    #     except IndexError:
    #         stop = None
    # # step
    # if len(str_slice.split(':')) == 3:
    #     step = int(str_slice.split(':')[-1])
    # else:
    #     step = None
    # return slice(start, stop, step)


def get_token_count(s: str, model_name: str = 'gpt-3.5-turbo') -> int:
    from tiktoken import encoding_for_model

    enc = encoding_for_model(model_name)
    tokenized_text = enc.encode(s)

    # calculate the number of tokens in the encoded text
    return len(tokenized_text)


def get_word_count(s: str) -> int:
    return len(s.split())


def format_date(dt: 'datetime') -> str:
    return dt.strftime('%Y-%m-%d')


def pymupdf_doc_page_info(document: 'Document') -> str:
    metadata = document.metadata
    total_pages_in_metadata = 'total_pages' in metadata
    if 'page_number' in metadata and total_pages_in_metadata:
        return f', Page {metadata["page_number"]}/{metadata["total_pages"]}'
    elif 'page' in metadata and total_pages_in_metadata:
        return f', Page {metadata["page"] + 1}/{metadata["total_pages"]}'
    else:
        return ''


def html_source_info(document: 'Document') -> str:
    metadata = document.metadata
    if 'source' in metadata:
        return f', Title: {metadata["title"]}, Source: {metadata["source"]}'
    else:
        return ''


def general_document_source_info(document: 'Document') -> str:
    metadata = document.metadata
    if 'source' in metadata:
        return f', Source: {metadata["source"]}'
    else:
        return ''


def save_str_to_tempfile(s: str, suffix: str = '.txt') -> str:
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(s)
        return f.name


def open_file(path: str):
    # if on macOS, open with default app
    if sys.platform == 'darwin':
        import subprocess

        subprocess.call(('open', path))
    # if on Linux, open with default app
    elif sys.platform.startswith('linux'):
        import subprocess

        subprocess.call(('xdg-open', path))
    # if on Windows, open with default app
    elif sys.platform == 'win32':
        import os

        os.startfile(path)
    else:
        raise NotImplementedError(f'Unsupported platform: {sys.platform}')


def deliver_prompts(
    what: str,
    documents: list['Document'],
    should_be_only_one_doc: bool = False,
    needs_splitting: bool = False,
    copy: bool = True,
    edit: bool = False,
    chunk_size: int = 2000,
    extra_chunk_info_fn: Callable[['Document'], str] = lambda doc: '',
    dry_run: bool = False,
    parts: list[int] | None = None,
    raw_triple_quotes: bool = False,
    raw: bool = False,
):
    from langchain.prompts import PromptTemplate

    def deliver_single_doc(document: 'Document'):
        if raw:
            template = RAW_TEMPLATE
        elif raw_triple_quotes:
            template = RAW_TRIPLE_QUOTES_TEMPLATE
        else:
            template = REPLY_OK_IF_YOU_READ_TEMPLATE
        prompt = PromptTemplate.from_template(template)
        content = document.page_content
        if raw or raw_triple_quotes:
            formatted_prompt = prompt.format(content=content)
        else:
            formatted_prompt = prompt.format(what=what, content=content)

        def edit_prompt(formatted_prompt: str = formatted_prompt):
            formatted_prompt_path = save_str_to_tempfile(
                formatted_prompt, suffix='.txt'
            )
            open_file(formatted_prompt_path)
            print(
                f'Please edit the prompt at {formatted_prompt_path} and copy it yourself.',
                file=sys.stderr,
            )
            return

        if edit and not dry_run:
            edit_prompt()
            return
        if copy:
            print(
                f'Word Count: {get_word_count(formatted_prompt)}, Char count: {len(formatted_prompt)}{extra_chunk_info_fn(document)}',
                file=sys.stderr,
            )
            if not dry_run:
                import pyperclip

                pyperclip.copy(formatted_prompt)
                print('Prompt copied to clipboard.', file=sys.stderr)
        else:
            print(formatted_prompt)

    def deliver_multiple_docs(documents: list['Document']):
        if len(documents) == 1:
            deliver_single_doc(documents[0])
            return
        if edit:
            print(f'Please copy the prompts after each edits.', file=sys.stderr)
        for i, doc in enumerate(documents):
            num_chunks = len(documents)
            if raw or raw_triple_quotes:
                if raw:
                    template = RAW_TEMPLATE
                else:
                    template = RAW_TRIPLE_QUOTES_TEMPLATE
                prompt = PromptTemplate.from_template(template)
            elif i == 0:
                prompt = PromptTemplate.from_template(
                    REPLY_OK_IF_YOU_READ_TEMPLATE_SPLITTED_FIRST
                )
            else:
                prompt = PromptTemplate.from_template(
                    REPLY_OK_IF_YOU_READ_TEMPLATE_SPLITTED_CONTINUED
                ).partial(x=str(i + 1))
            content = doc.page_content
            if raw or raw_triple_quotes:
                formatted_prompt = prompt.format(content=content)
            else:
                formatted_prompt = prompt.format(what=what, content=content)
            if dry_run:
                print(
                    f'Press Enter to copy prompt {i+1}/{num_chunks}. Word Count: {get_word_count(formatted_prompt)}, Char count: {len(formatted_prompt)}{extra_chunk_info_fn(doc)}: '
                )
                continue
            if edit:
                input(
                    f'Press Enter to edit prompt {i+1}/{num_chunks}. Word Count: {get_word_count(formatted_prompt)}, Char count: {len(formatted_prompt)}{extra_chunk_info_fn(doc)}: '
                )
                formatted_prompt_path = save_str_to_tempfile(
                    formatted_prompt, suffix='.txt'
                )
                open_file(formatted_prompt_path)
                continue
            input(
                f'Press Enter to copy prompt {i+1}/{num_chunks}. Word Count: {get_word_count(formatted_prompt)}, Char count: {len(formatted_prompt)}{extra_chunk_info_fn(doc)}: '
            )
            import pyperclip

            pyperclip.copy(formatted_prompt)

    if dry_run:
        print(
            f'Dry running. Nothing will be copied to your clipboard, and you don\''
            't need to press Enter to move forward.'
        )
    if needs_splitting:
        from langchain.text_splitter import TokenTextSplitter

        splitter = TokenTextSplitter(encoding_name='cl100k_base', chunk_size=chunk_size)
        splitted = splitter.split_documents(documents)
        if parts:
            len_splitted = len(splitted)
            parts = list({part for part in parts if 0 <= part - 1 < len_splitted})
            print(
                f'Selecting {len(parts)} parts out of {len_splitted}.', file=sys.stderr
            )
            print(f'Using parts: {parts}', file=sys.stderr)
            splitted = [splitted[i - 1] for i in parts]
        deliver_multiple_docs(splitted)

    elif should_be_only_one_doc:
        deliver_single_doc(documents[0])
    else:
        deliver_multiple_docs(documents)


def assert_never(a: NoReturn) -> NoReturn:
    raise RuntimeError("Should not get here")


def save_stdin_to_tempfile() -> str:
    # create a temp file and save stdin to it, and return the tempfile path
    import tempfile
    import shutil
    import sys

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        with open(temp_file.name, 'w') as f:
            shutil.copyfileobj(sys.stdin, f)
        temp_file_path = temp_file.name
    return temp_file_path


def save_clipboard_to_tempfile() -> str:
    # create a temp file and save stdin to it, and return the tempfile path
    import tempfile
    import pyperclip
    import sys

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        with open(temp_file.name, 'w') as f:
            f.write(pyperclip.paste())
        temp_file_path = temp_file.name
    return temp_file_path


def get_percentage_non_ascii(s: str) -> float:
    return sum(1 for c in s if ord(c) >= 128) / len(s)


def get_default_chunk_size(model: str | None = None) -> int:
    from langchain_utils.config import MODEL_TO_CONTEXT_LENGTH_MAPPING, DEFAULT_MODEL

    if model not in MODEL_TO_CONTEXT_LENGTH_MAPPING:
        model = DEFAULT_MODEL
    return MODEL_TO_CONTEXT_LENGTH_MAPPING[model] // 2


def extract_github_info(url: str) -> dict[str, str] | None:
    import re

    # Define a regular expression to match GitHub URLs
    pattern = (
        r"^https?://github\.com/([^/]+)/([^/]+)(?:/(?:tree|blob)/([^/]+)(?:/(.+))?)?$"
    )

    # Use the regular expression to extract the URL components
    match = re.match(pattern, url)
    if match:
        repo_owner = match.group(1)
        repo_name = match.group(2)
        revision = (
            match.group(3) or "master"
        )  # Use "main" as the default revision if not provided
        file_path = (
            match.group(4) or "README.md"
        )  # Use "README.md" as the default file path if not provided

        return {
            "repo_owner": repo_owner,
            "repo_name": repo_name,
            "revision": revision,
            "file_path": file_path,
        }

    return None


def get_github_file_raw_url(
    repo_owner: str,
    repo_name: str,
    revision: str = 'master',
    file_path: str = 'README.md',
):
    # Construct the raw URL for the README.md file
    raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{revision}/{file_path}"

    return raw_url
