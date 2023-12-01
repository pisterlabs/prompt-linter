import logging
import os
import time
from glob import glob
from pathlib import Path

from yaspin import yaspin
from yaspin.spinners import Spinners

spinner = yaspin(Spinners.dots, color="blue")

with spinner:
    from dotenv import load_dotenv

    # Remove error message from missing azure.core import
    logging.getLogger().setLevel(logging.ERROR)

    from langchain.chat_models import ChatOpenAI

    logging.getLogger().setLevel(logging.WARNING)

    import sys
    import traceback

    import inquirer
    from llama_index import (
        GPTVectorStoreIndex,
        LLMPredictor,
        ServiceContext,
        StorageContext,
        download_loader,
        load_index_from_storage,
        QuestionAnswerPrompt,
    )
    from llama_index.prompts.default_prompts import DEFAULT_REFINE_PROMPT

load_dotenv()

llm_predictor = LLMPredictor(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

PATH_TO_PDFS = "pdfs"
PATH_TO_INDEXES = "gpt_indexes"


# build index from PDF
def pdf_to_index(pdf_path, save_path):
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    documents = loader.load_data(file=Path(pdf_path))
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=save_path)
    print("\033[0;32mSaved PDF index to disk\033[0m")


# query index using GPT
def query_index(query_u, pdf_name):
    QA_PROMPT_TMPL = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information above, please answer the question clearly but as concisely as possible: {query_str}\n"
    )
    QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

    storage_context = StorageContext.from_defaults(
        persist_dir=f"{PATH_TO_INDEXES}/{pdf_name}"
    )
    index = load_index_from_storage(storage_context, service_context=service_context)
    query_engine = index.as_query_engine(streaming=True, text_qa_template=QA_PROMPT)
    response = query_engine.query(query_u)

    return response


def save_pdf(file_path, absolute=False):
    _, file_name = os.path.split(file_path)

    if absolute:
        pdf_to_index(
            pdf_path=f"{file_path}", save_path=f"{PATH_TO_INDEXES}/{file_name}"
        )
    else:
        pdf_to_index(
            pdf_path=f"{PATH_TO_PDFS}/{file_name}",
            save_path=f"{PATH_TO_INDEXES}/{file_name}",
        )


if __name__ == "__main__":
    if not os.path.exists(PATH_TO_INDEXES):
        os.makedirs(PATH_TO_INDEXES)
    if not os.path.exists(PATH_TO_PDFS):
        os.makedirs(PATH_TO_PDFS)

    try:
        while True:
            menu_questions = [
                inquirer.List(
                    "menu_choice",
                    message="What would you like to do?",
                    choices=["Query a document", "Add a new document", "Quit"],
                    carousel=True,
                ),
            ]

            menu_answers = inquirer.prompt(
                menu_questions, raise_keyboard_interrupt=True
            )
            menu_choice = menu_answers.get("menu_choice")

            if menu_choice == "Quit":
                break
            elif menu_choice == "Query a document":
                dirs = [
                    d
                    for d in os.listdir(PATH_TO_INDEXES)
                    if os.path.isdir(os.path.join(PATH_TO_INDEXES, d))
                ]
                if len(dirs) == 0:
                    print("\033[0;31mNo PDFs were found\033[0m")
                    continue

                query_doc_questions = [
                    inquirer.List(
                        "query_doc_choice",
                        message="Select a document to query",
                        choices=dirs,
                        carousel=True,
                    ),
                ]

                query_doc_answer = inquirer.prompt(
                    query_doc_questions, raise_keyboard_interrupt=True
                )
                query_doc_choice = query_doc_answer.get("query_doc_choice")

                while True:
                    query = input("\033[0;33m> ")
                    print("\033[0m", end="")

                    if query == "exit":
                        break
                    elif not query or not query.strip():
                        continue

                    with spinner:
                        res = query_index(query_u=query, pdf_name=query_doc_choice)

                    print("\033[0;36m", end="")
                    res.print_response_stream()
                    print("\033[0m\n")
            elif menu_choice == "Add a new document":
                docs = glob(os.path.join(PATH_TO_PDFS, "*.pdf"))
                docs.insert(0, "Enter the path to a PDF")
                add_doc_questions = [
                    inquirer.List(
                        "add_doc_choice",
                        message="Select a document to add",
                        choices=docs,
                        carousel=True,
                    ),
                ]

                add_doc_answer = inquirer.prompt(
                    add_doc_questions, raise_keyboard_interrupt=True
                )
                add_doc_choice = add_doc_answer.get("add_doc_choice")

                if add_doc_choice == "Enter the path to a PDF":
                    doc_path = inquirer.prompt(
                        [inquirer.Path("doc_path")], raise_keyboard_interrupt=True
                    ).get("doc_path")

                    if not os.path.isfile(doc_path):
                        print("\033[0;31mUnable to find the document.\033[0m")
                    else:
                        with spinner:
                            save_pdf(doc_path, absolute=True)
                        print("Added your PDF")
                else:
                    with spinner:
                        save_pdf(add_doc_choice)
                    print("Added your PDF")
    except KeyboardInterrupt:
        print("\033[0m")
        sys.exit()
    except Exception as e:
        traceback.print_exc()
    finally:
        print("\033[0m")
