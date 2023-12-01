
import os
import json
from argparse import ArgumentParser
from glob import glob
from time import time

from tqdm import tqdm
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS

from util.modelhelpers import get_llm_model, get_embeddings_model, text_splitter


# load environment variables from .env, tokens
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')


def get_argparser():
    argparser = ArgumentParser(description='Learn the context.')
    argparser.add_argument('corpusdir', help='directory of the PDF books')
    argparser.add_argument('learnconfigpath', help='path of the JSON file with the configs')
    argparser.add_argument('outputdir', help='target directory')
    return argparser


def get_pages_from_pdf_document(pdffilepath):
    loader = PyPDFLoader(pdffilepath)
    pages = loader.load_and_split(text_splitter=text_splitter)
    return pages


def get_pages_from_pdf_documents(directory):
    pages = []
    for pdffilepath in tqdm(glob(os.path.join(directory, '*.pdf'))):
        this_pages = get_pages_from_pdf_document(pdffilepath)
        if this_pages is not None and len(this_pages) > 0:
            pages += this_pages
    return pages


def iterate_list_pdfnames(directory):
    for pdffilepath in tqdm(glob(os.path.join(directory, '*.pdf'))):
        basename = os.path.basename(pdffilepath)
        yield basename


def generate_model_and_faissdb(corpusdir, config):
    llm = get_llm_model(config)
    embedding = get_embeddings_model(config)

    pages = get_pages_from_pdf_documents(corpusdir)
    db = FAISS.from_documents(pages, embedding)


    return llm, embedding, db


if __name__ == '__main__':
    args = get_argparser().parse_args()
    config = json.load(open(args.learnconfigpath, 'r'))
    if not os.path.isdir(args.outputdir):
        raise FileNotFoundError('Output directory {} does not exist.'.format(args.outputdir))

    starttime = time()
    print('Generating FAISS...')
    _, _, db = generate_model_and_faissdb(args.corpusdir, config)
    indextime = time()
    print('Finished. (Duration: {:.2f} s)'.format(indextime-starttime))
    print("=======")
    print('Saving FAISS...')
    db.save_local(args.outputdir)
    savetime = time()
    print('Saved. (Duration: {:.2f} s)'.format(savetime-indextime))
    print("=======")
    print('Saving other configurations...')
    json.dump(config, open(os.path.join(args.outputdir, 'config.json'), 'w'))
    with open(os.path.join(args.outputdir, 'booklist.txt'), 'w') as f:
        for bookname in iterate_list_pdfnames(args.corpusdir):
            f.write(bookname+'\n')
    endtime = time()
    print('Saved. (Duration: {:.2f} s)'.format(endtime-savetime))

    print('Total duration: {:.2f} s'.format(endtime-starttime))
