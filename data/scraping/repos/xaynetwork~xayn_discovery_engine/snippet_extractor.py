# Copyright 2023 Xayn AG
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from typing import List, Callable
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TextSplitter,
    NLTKTextSplitter,
)
from transformers import PreTrainedTokenizerFast
from msgpack import Packer, Unpacker


# like `langchain.text_splitter.NLTKTextSplitter` but with configurable language and not small split merging
class NLTKTextSplitter(TextSplitter):
    """Splitting text using NLTK package."""

    def __init__(self, *, language, separator="\n\n", **kwargs):
        """Initialize the NLTK splitter."""
        super().__init__(**kwargs)
        self._tokenizer = lambda x: sent_tokenize(x, language)
        self._separator = separator

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        splits = self._tokenizer(text)
        return self._merge_splits(splits, self._separator)


class TextSplitterWithBigChunkSplitter(TextSplitter):
    def __init__(
        self,
        *,
        primary: TextSplitter,
        secondary: TextSplitter,
        hard_chunk_size_limit: int,
        length_function: Callable[[str], int]
    ):
        # setting chunk_size = hard_max_chunk is needed for using self._merge_splits
        super().__init__(
            chunk_size=hard_chunk_size_limit,
            chunk_overlap=0,
            length_function=length_function,
        )
        self._primary = primary
        self._secondary = secondary
        self._hard_chunk_size_limit = hard_chunk_size_limit

    def split_text(self, text: str) -> List[str]:
        main_splits = self._primary.split_text(text)

        # remove snippets that are larger than hard_max_chunk
        no_big_splits = []
        for split in main_splits:
            if self._length_function(split) > self._hard_chunk_size_limit:
                secondary_splits = self._secondary.split_text(split)
                no_big_splits.extend(secondary_splits)
            else:
                no_big_splits.append(split)

        return self._merge_splits(no_big_splits, "\n")


class SnippetExtractor(TextSplitterWithBigChunkSplitter):
    def __init__(
        self,
        *,
        language: str,
        chunk_size: int,
        hard_chunk_size_limit: int,
        tokenizer,
    ):
        token_len = lambda s: len(tokenizer(s).input_ids)
        super().__init__(
            primary=NLTKTextSplitter(
                language=language,
                chunk_size=chunk_size,
                chunk_overlap=0,
                length_function=token_len,
            ),
            secondary=RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=0, length_function=token_len
            ),
            hard_chunk_size_limit=hard_chunk_size_limit,
            length_function=token_len,
        )

def ok(value: any) -> dict:
    return { 'Ok': value }
def err(value: any) -> dict:
    return { 'Err': str(value) }


def run_stdio_client():
    import sys
    tokenizers = {}

    # There is some bad interaction between stdin Binary I/O buffering code
    # and the Unpacker with can lead to hangs. Using Raw I/O avoids this issue.
    stdin = sys.stdin.buffer.raw
    stdout = sys.stdout.buffer

    # Disable buffer size safety check, we already checked it on the rust side
    # and keeping that option is sync is another potential source of bugs.
    unpacker = Unpacker(stdin, max_buffer_size = 0)
    packer = Packer()

    stdout.write(packer.pack("ready"));
    stdout.flush()

    for msg in unpacker:
        result = None
        try:
            tag = msg['tag']
            cmd = msg['cmd']
            if tag == 'initialize_tokenizer':
                tokenizers[cmd['name']] = PreTrainedTokenizerFast(tokenizer_file=cmd['path'])
                result = ok(True)
            elif tag == 'extract':
                snippets = SnippetExtractor(
                    language = cmd['language'],
                    chunk_size = cmd['chunk_size'],
                    hard_chunk_size_limit = cmd['hard_chunk_size_limit'],
                    tokenizer = tokenizers[cmd['tokenizer']],
                ).split_text(cmd['document'])
                result = ok(snippets)
            elif tag == 'ping':
                result = ok(True)
            else:
                result = err(f"unknown command: {tag}")
        except Exception as error:
            result = err(error)
        finally:
            stdout.write(packer.pack(result))
            stdout.flush()

if __name__ == '__main__':
    run_stdio_client()
