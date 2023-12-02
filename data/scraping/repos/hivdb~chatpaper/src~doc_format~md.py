from collections import defaultdict
import re
from pathlib import Path
from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


# TODO, split paragraph
# TODO, adjust the header level if H4 after H2 and H3 is missing


def split_md_section(markdownfile: Path):

    with open(markdownfile) as fd:
        content = fd.readlines()

    content = group_by_headers(content)
    content = get_header_level(content)

    for c in content:
        c['paragraphs'] = get_paragraph_list(c)

    return content


def get_paragraph_list(one_section):
    paragraphs = [
        one_section['header']
    ]
    new_paragraph = []

    for i in one_section['content']:
        if i.strip() == '' and new_paragraph:
            paragraphs.append(''.join(new_paragraph))
            new_paragraph = []
            continue

        if i.strip() != '':
            new_paragraph.append(i)

    if new_paragraph:
        paragraphs.append(''.join(new_paragraph))

    # assert same content
    old = ''.join([one_section['header']] + [
        i
        for i in one_section['content']
        if i.strip()
    ])

    new = ''.join([
        i
        for i in paragraphs
    ])

    assert (old == new)

    return paragraphs


def group_by_headers(lines):
    # TODO check the markdown file is well-formatted, starting with blank line
    # or starting with highest header level

    group_by_headers = []

    section_header = None
    section_content = None
    section_order = 0

    for line_no, line in enumerate(lines):
        # prev_line = lines[line_no-1] if line_no > 0 else ''
        if not re.search(r'^\#+\s*[^\#]+', line):  # and prev_line.strip() == '':
            section_content.append(line)
            continue

        if section_header is not None:
            group_by_headers.append({
                'content': section_content,
                'text_order': section_order,
                'header': section_header,
                'all_content': f"{section_header}{''.join(section_content)}"
            })

        section_header = line
        section_content = []
        section_order += 1

    if section_content:
        group_by_headers.append({
                'content': section_content,
                'text_order': section_order,
                'header': section_header,
                'all_content': f"{section_header}{''.join(section_content)}"
            })

    return group_by_headers


def get_header_level(sections):

    for content in sections:
        header_level = _get_header_level(content['header'])
        content['level'] = header_level

    return sections


def _get_header_level(header):
    level = 0
    for i in header:
        if i != '#':
            break
        level += 1

    return level


class MarkdownSectionLoader(BaseLoader):

    def __init__(self, file_path: Path):
        self.file_path = file_path

    def load(self) -> List[Document]:
        docs = [
            i['all_content']
            for i in split_md_section(self.file_path)
        ]

        metadata = {"source": str(self.file_path)}
        return [
            Document(page_content=i, metadata=metadata)
            for i in docs
        ]


class MarkdownParagraphLoader(BaseLoader):

    def __init__(self, file_path: Path):
        self.file_path = file_path

    def load(self) -> List[Document]:
        docs = [
            j
            for i in split_md_section(self.file_path)
            for j in i['paragraphs']
        ]

        metadata = {"source": str(self.file_path)}
        return [
            Document(page_content=i, metadata=metadata)
            for i in docs
        ]
