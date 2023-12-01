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
    pymupdf_doc_page_info,
    convert_str_slice_notation_to_slice,
    get_percentage_non_ascii,
    get_token_count,
    get_default_chunk_size,
)
from langchain_utils.loaders import load_pdf
from langchain_utils.config import DEFAULT_PDF_WHAT, TESSERACT_OCR_DEFAULT_LANG
from langchain_utils.utils_argparse import get_get_prompt_base_arg_parser


def get_args():
    """Get command-line arguments"""

    parser = get_get_prompt_base_arg_parser(
        description='Get a prompt consisting the text content of a PDF file'
    )

    parser.add_argument(
        'pdf_path', help='Path to the PDF file', metavar='PDF Path', type=str
    )
    parser.add_argument(
        '-p',
        '--pages',
        help='Only include specified page numbers',
        type=int,
        nargs='+',
        default=None,
    )
    parser.add_argument(
        '-l',
        '--page-slice',
        help='Use Python slice syntax to select page numbers (e.g. 1:3, 1:10:2, etc.)',
        type=str,
        default=None,
    )
    parser.add_argument(
        '-M',
        '--merge',
        help='Merge contents of all pages before processing',
        action='store_true',
    )
    parser.add_argument(
        '-w',
        '--what',
        help='Initial knowledge you want to insert before the PDF content in the prompt',
        type=str,
        default=DEFAULT_PDF_WHAT,
    )
    parser.add_argument(
        '-o',
        '--fallback-ocr',
        help='Use OCR as fallback if no text detected on page, please set TESSDATA_PREFIX environment variable to the path of your tesseract data directory',
        action='store_true',
    )
    parser.add_argument(
        '-L',
        '--ocr-language',
        help='Language to use for Tesseract OCR',
        type=str,
        default=TESSERACT_OCR_DEFAULT_LANG,
    )

    args = parser.parse_args()
    args.chunk_size = get_default_chunk_size(args.model)
    return args


def main():
    """Make a jazz noise here"""

    args = get_args()

    print(f'Loading PDF from {args.pdf_path} ...', file=sys.stderr)
    if args.fallback_ocr:
        docs = load_pdf(
            args.pdf_path,
            use_ocr_if_no_text_detected_on_page=True,
            ocr_language=args.ocr_language,
        )
    else:
        docs = load_pdf(args.pdf_path)
    num_whole_pdf_pages = len(docs)
    if args.pages and args.page_slice:
        print(
            'Please specify either --pages or --page-slice, not both',
            file=sys.stderr,
        )
        sys.exit(1)
    if args.pages:
        args.pages = [p - 1 for p in args.pages if p <= num_whole_pdf_pages and p > 0]
        docs = [doc for doc in docs if doc.metadata['page'] in args.pages]
    if args.page_slice:
        args.pages = list(
            x - 1
            for x in list(range(num_whole_pdf_pages))[
                convert_str_slice_notation_to_slice(args.page_slice)
            ]
        )
        docs = [doc for doc in docs if doc.metadata['page'] in args.pages]
    texts = [doc.page_content for doc in docs]
    all_text = '\n'.join(texts)
    word_count = get_word_count((all_text))
    char_count = len(all_text)
    print(
        f'Loaded {len(docs)} pages. Word count: {word_count} Char count: {len(all_text)}',
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
    if args.merge or args.no_split:
        from langchain.docstore.document import Document

        merged = Document(
            page_content=all_text,
            metadata={
                k: v for k, v in docs[0].metadata.items() if k not in {'page_number'}
            },
        )
    if args.no_split:
        needs_splitting = False
    elif word_count > args.chunk_size * 0.75:
        needs_splitting = True
    else:
        needs_splitting = False
    deliver_prompts(
        what=args.what,
        documents=[merged] if args.merge or args.no_split else docs,  # type: ignore
        needs_splitting=needs_splitting,
        copy=args.copy,
        edit=args.edit,
        chunk_size=args.chunk_size,
        extra_chunk_info_fn=pymupdf_doc_page_info,
        dry_run=args.dry_run,
        raw_triple_quotes=args.raw,
        raw=args.raw_no_quotes,
        parts=args.parts,
    )


if __name__ == '__main__':
    main()
