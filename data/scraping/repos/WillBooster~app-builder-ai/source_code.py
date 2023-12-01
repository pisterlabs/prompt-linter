import difflib
import logging
import os

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import load_prompt
from langchain.vectorstores import FAISS
from parsers.code_output_parser import CodeOutputParser
from schema import PublicInterfaceDocument
from util import (
    ACCEPTANCE_TEST_PREFIX,
    SRC_DIR,
    UNIT_TEST_PREFIX,
    execute_model,
    get_prompt_file_path,
    get_src_file_path,
    get_unit_test_file_name,
)


def generate_source_code(
    model: ChatOpenAI,
    specifications_text: str,
    public_interface_document: PublicInterfaceDocument,
) -> None:
    for file in public_interface_document.files:
        if os.path.exists(get_src_file_path(file.name)):
            logging.info(f"Reusing {file.name}.")
            continue

        logging.info(f"Generating {file.name}.")

        if file.name == public_interface_document.entry_point_file_name:
            test_code = "No test for this file."
        else:
            test_file_name = get_unit_test_file_name(file.name)
            test_code = open(get_src_file_path(test_file_name)).read()

        output_parser = CodeOutputParser()
        prompt = load_prompt(get_prompt_file_path("gen_source_code.yaml")).format(
            specifications=specifications_text,
            public_interface_document=public_interface_document.json(),
            test_code=test_code,
            format_instructions=output_parser.get_format_instructions(),
            file=file.name,
        )
        output = execute_model(model, prompt)
        source_code = output_parser.parse(output)

        with open(get_src_file_path(file.name), "w") as f:
            f.write(source_code)


def create_source_code_vector_db() -> FAISS:
    logging.info("Creating source code vector database.")

    loader = DirectoryLoader(
        SRC_DIR,
        glob="**/*.py",
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader,
    )
    docs = loader.load()
    docs = [
        doc
        for doc in docs
        if UNIT_TEST_PREFIX not in doc.metadata["source"]
        and ACCEPTANCE_TEST_PREFIX not in doc.metadata["source"]
    ]
    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    return db


def modify_source_code(
    model: ChatOpenAI,
    specifications_text: str,
    change_request: str,
    file_name: str,
    public_interface_document: PublicInterfaceDocument,
) -> bool:
    logging.info(f"Modifying source code for {file_name}.")

    with open(get_src_file_path(file_name), "r") as f:
        source_code = f.read()

    output_parser = CodeOutputParser()
    prompt = load_prompt(get_prompt_file_path("modify_source_code.yaml")).format(
        specifications=specifications_text,
        change_request=change_request,
        public_interface_document=public_interface_document.json(),
        format_instructions=output_parser.get_format_instructions(),
        file=file_name,
        source_code=source_code,
    )
    output = execute_model(model, prompt)
    fixed_source_code = output_parser.parse(output)

    if fixed_source_code.strip() == "":
        logging.info(f"No changes to source code for {file_name}.")
        return False
    else:
        with open(get_src_file_path(file_name), "w") as f:
            f.write(fixed_source_code)
        logging.info(f"Modified source code for {file_name}.")

        diff = difflib.unified_diff(
            source_code.splitlines(), fixed_source_code.splitlines(), lineterm=""
        )
        print("\n".join(diff))

        return True
