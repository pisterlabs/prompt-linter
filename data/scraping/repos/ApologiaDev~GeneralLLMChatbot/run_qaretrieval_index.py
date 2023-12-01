
import os
import json
from argparse import ArgumentParser

from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from util.modelhelpers import get_llm_model, get_embeddings_model


# load environment variables from .env, tokens
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')


def get_argparser():
    argparser = ArgumentParser(description='Q & A Retrieval from the context.')
    argparser.add_argument('contextdir', help='directory for the saved context')
    return argparser


def load_faiss(contextdir, embedding):
    db = FAISS.load_local(contextdir, embedding)
    return db


if __name__ == '__main__':
    args = get_argparser().parse_args()
    if not os.path.isdir(args.contextdir):
        raise FileNotFoundError('Directory {} not found.'.format(args.contextdir))
    config = json.load(open(os.path.join(args.contextdir, 'config.json')))
    llm = get_llm_model(config)
    embedding = get_embeddings_model(config)
    db = load_faiss(args.contextdir, embedding)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    done = False
    while not done:
        inputtext = input('Prompt> ')
        if len(inputtext.strip()) == 0:
            done = True
        else:
            reply = qa({'query': inputtext})
            print(reply['result'])
            print("=====")
