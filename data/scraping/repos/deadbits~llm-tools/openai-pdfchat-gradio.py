#!/usr/bin/env python3
##
# Chat with PDF document using OpenAI, LlamaIndex, and Gradio
# github.com/deadbits
##
import os
import sys
import logging
import argparse
import openai
import gradio as gr
import urllib.request

from pathlib import Path

from langchain import OpenAI

from llama_index import GPTSimpleVectorIndex
from llama_index import LLMPredictor
from llama_index import ServiceContext
from llama_index import download_loader


# disable some of the more verbose llama_index logging
logging.basicConfig(level=logging.CRITICAL)

# change this to your preferred model
MODEL_NAME = 'gpt-3.5-turbo'


def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def load_document(fpath):
    print('[status] loading document ({})'.format(fpath))
    PDFReader = download_loader('PDFReader')
    loader = PDFReader()
    docs = loader.load_data(file=Path(fpath))
    return docs


def answer_question(url='', fpath='', question='', api_key=''):
    if url.strip() == '' and fpath is None:
        return '[error] file and url cannot both be empty'

    if url.strip() != '' and fpath is not None:
        return '[error] file and url cannot both be provided'

    if question.strip() == '':
        return '[error] question cannot be empty'

    if api_key.strip() == '':
        return '[error] OpenAI API key cannot be empty'

    if url.strip() != '':
        download_pdf(url, 'corpus.pdf')
        fpath = 'corpus.pdf'
    elif fpath != '':
        fname = fpath.name
        os.rename(fname, 'corpus.pdf')

    docs = load_document('corpus.pdf')

    llm = LLMPredictor(llm=OpenAI(openai_api_key=api_key, temperature=0, model_name=MODEL_NAME))
    ctx = ServiceContext.from_defaults(llm_predictor=llm, chunk_size_limit=1024)
    index = GPTSimpleVectorIndex.from_documents(docs, service_context=ctx)

    response = index.query(question)
    response = str(response)
    if response.startswith('\n'):
        response = response[1:]

    return response


title = 'PDF Chat with OpenAI'
description = """Upload local PDF document or enter URL to PDF"""

with gr.Blocks() as demo:

    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(description)

    with gr.Row():

        with gr.Group():
            openai_api_key = gr.Textbox(label='OpenAI API key')

            url = gr.Textbox(label='PDF URL to download')
            gr.Markdown("<center><h4>OR<h4></center>")
            fpath = gr.File(label='Upload PDF', file_types=['.pdf'])

            question = gr.Textbox(label='User prompt')
            btn = gr.Button(value='Submit')
            btn.style(full_width=True)

        with gr.Group():
            answer = gr.Textbox(label='Response:', lines=15, placeholder='Output')

        btn.click(answer_question, inputs=[url, fpath, question, openai_api_key], outputs=[answer])

demo.launch(share=True)
