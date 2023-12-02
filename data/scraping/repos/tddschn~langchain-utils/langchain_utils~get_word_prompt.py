#!/usr/bin/env python3
"""
Author : Xinyuan Chen <45612704+tddschn@users.noreply.github.com>
Date   : 2023-04-09
"""

import sys

from langchain_utils import __version__
from langchain_utils.utils import (
    deliver_prompts,
    get_word_count,
    deliver_prompts,
    general_document_source_info,
    get_percentage_non_ascii,
    get_token_count,
    get_default_chunk_size,
)
from langchain_utils.loaders import load_word, UnstructuredLoadingMode
from langchain_utils.config import DEFAULT_GENERAL_WHAT
from langchain_utils.utils_argparse import get_get_prompt_base_arg_parser


def get_args():
    """Get command-line arguments"""

    parser = get_get_prompt_base_arg_parser(description='Get a prompt from text files')

    parser.add_argument(
        'word_path',
        help='Path to the Word document',
        metavar='Word Document Path',
        type=str,
    )
    parser.add_argument(
        '-u', '--unstructured-loading-mode', type=str, choices=UnstructuredLoadingMode.__args__, default='single', help='Unstructured loading mode'  # type: ignore
    )

    parser.add_argument(
        '-w',
        '--what',
        help='Initial knowledge you want to insert before the PDF content in the prompt',
        type=str,
        default=DEFAULT_GENERAL_WHAT,
    )
    parser.add_argument(
        '-M',
        '--merge',
        help='Merge contents of all pages before processing',
        action='store_true',
    )

    args = parser.parse_args()
    args.chunk_size = get_default_chunk_size(args.model)
    return args


def main():
    """Make a jazz noise here"""

    args = get_args()

    print(f'Loading Word document from {args.word_path} ...', file=sys.stderr)
    docs = load_word(args.word_path)
    texts = [doc.page_content for doc in docs]
    all_text = '\n'.join(texts)
    word_count = get_word_count((all_text))
    char_count = len(all_text)
    print(
        f'Loaded {len(docs)} Documents. Word count: {word_count} Char count: {char_count}',
        file=sys.stderr,
    )
    if args.print_percentage_non_ascii:
        print(
            f'Percentage of non-ascii characters: {get_percentage_non_ascii(all_text) * 100:.2f}%',
            file=sys.stderr,
        )
        token_count = get_token_count(all_text)
        print(f'Token count: {token_count}', file=sys.stderr)
        print(f'Token / Word: {token_count / word_count:.2f}', file=sys.stderr)
        print(f'Token / Char: {token_count / char_count:.2f}', file=sys.stderr)
        return
    if args.merge:
        from langchain.docstore.document import Document

        merged = Document(
            page_content=all_text,
            metadata={k: v for k, v in docs[0].metadata.items() if k in {'source'}},
        )
    documents = [merged] if args.merge else docs  # type: ignore
    num_docs = len(documents)
    if args.no_split:
        needs_splitting = False
    elif word_count > args.chunk_size * 0.75:
        needs_splitting = True
    else:
        needs_splitting = False
    deliver_prompts(
        what=args.what,
        documents=documents,  # type: ignore
        needs_splitting=needs_splitting,
        copy=args.copy,
        edit=args.edit,
        should_be_only_one_doc=True if num_docs == 1 else False,
        chunk_size=args.chunk_size,
        extra_chunk_info_fn=general_document_source_info,
        dry_run=args.dry_run,
        raw_triple_quotes=args.raw,
        raw=args.raw_no_quotes,
        parts=args.parts,
    )


if __name__ == '__main__':
    main()
