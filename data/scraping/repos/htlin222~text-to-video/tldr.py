# -*- coding: utf-8 -*-
import openai
import wget
import pathlib
import pdfplumber
import numpy as np
import sys
import re
from pathlib import Path

def getPaper(paper_url, filename="random_paper.pdf"):
    """
    Downloads a paper from it's arxiv page and returns
    the local path to that file.
    """
    downloadedPaper = wget.download(paper_url, filename)
    downloadedPaperFilePath = pathlib.Path(downloadedPaper)

    return downloadedPaperFilePath


def showPaperSummary(paperContent):
    tldr_tag = "\n tl;dr:"
    openai.organization = 'API KEY org'
    openai.api_key = "your openAI key"
    engine_list = openai.Engine.list()

    for page in paperContent:
        text = page.extract_text() + tldr_tag
        response = openai.Completion.create(engine="davinci",prompt=text,temperature=0.3,
            max_tokens=140,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )
        print(response["choices"][0]["text"])

# TODO: not done yet
if __name__=='__main__':
    my_file = Path(sys.argv[1])
    folder = re.sub(r'\.[A-Za-z]*','',sys.argv[1])
    if my_file.is_file():
        paperContent = pdfplumber.open(my_file).pages
        showPaperSummary(paperContent)
        print('\n✨TL;DR done')
    else:
        print('❌ please create file:', my_file, 'first, thank you.')
        file = open(sys.argv[1], 'w', encoding='utf-8')
