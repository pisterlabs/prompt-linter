import argparse
import glob
import logging
from os import path

from dotenv import load_dotenv
import langchain

from transist import index


SECTIONS = [
    "project details",
    "safeguards",
    "application of methodology",
    "quantification of ghg emission",
    "monitoring"
]


def search_index(pdd_index_dir, method_index_dir):
    pdd_coll = index.PddLlamaCollection(pdd_index_dir)
    method_coll = index.MethodologyCollection(method_index_dir)

    for question in [
        "what methodologies can be applied to projects involving cook stoves",
        "what methodologies can be applied to afforestation projects",
        "what are the requirements prescribed by methodology AMS.II-G for energy efficiency measures"
    ]:
        print("##### Methdology Queston #####")
        print(question)
        print("\n")
        # results = pdd_coll.search(question, n=5)
        # for result in results:
        #     print(f"""text: {result.text}\n\nmetadata: {result.metadata}\n\n\n""")

        answer = pdd_coll.answer(question)
        print("###### PDD ANSWER ######")
        print(answer)
        print("\n")

        # results = method_coll.search(question, n=5, mmr=True)
        # for result in results:
        #     print(f"""result: {result.text}\n\n metadata:{result.metadata}""")

        answer = method_coll.answer(question)
        print("###### METHODOLOGY ANSWER ######")
        print(answer)
        print("\n\n")


def make_pdd_index(pdd_dir, pdd_index_dir):
    if pdd_dir is not None:
        index.PddLlamaCollection.from_directory(pdd_dir, pdd_index_dir)


def make_methodology_index(method_dir, method_index_dir):
    if method_dir is not None:
        index.MethodologyCollection.from_directory(method_dir, method_index_dir)


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("transist").setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(
        prog="rag_index",
        description="Manipulate Transistry RAG index"
    )
    parser.add_argument("-i", "--index_dir", action="store", required=True)
    parser.add_argument("-p", "--pdd_dir", action="store", default=None, required=False)
    parser.add_argument("-m", "--method_dir", action="store", default=None, required=False)

    args = parser.parse_args()
    langchain.verbose = True

    if args.pdd_dir:
        make_pdd_index(pdd_dir=args.pdd_dir,
                       pdd_index_dir=path.join(args.index_dir, 'pdd'))
    if args.method_dir:
        make_methodology_index(method_dir=args.method_dir,
                               method_index_dir=path.join(args.index_dir, 'method'))

    search_index(path.join(args.index_dir, 'pdd'),
                 path.join(args.index_dir, 'method'))
