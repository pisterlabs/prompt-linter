import sys, os
import json
import shutil
import os, re
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chains.openai_functions import (
    create_structured_output_chain,
)

sys.path.append(os.environ.get("PROJECT_PATH"))
os.chdir(os.environ.get("PROJECT_PATH"))

import utils.custom_langchain as clc
import utils.paper_utils as pu
import utils.prompts as ps
import utils.db as db

data_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "arxiv_text")
meta_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "arxiv_meta")
qna_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "arxiv_qna")
chunk_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "arxiv_chunks")

## Helper splitter.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

## LLM model.
llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.1)

## Initialize chain.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ps.QNA_SYSTEM_PROMPT),
        ("user", ps.QNA_USER_PROMPT),
        (
            "user",
            "Tip: Remember to always include citations in your answers. Make sure your questions are detailed and stand alone. Do not reference the sample questions.",
        ),
    ]
)
parser = clc.CustomFixParser(pydantic_schema=ps.QnaSet)
chain = create_structured_output_chain(
    ps.QnaSet, llm, prompt, output_parser=parser, verbose=False
)


def main():
    """Convert papers into set of Q&A pairs."""
    ## Get raw paper list.
    fnames = os.listdir(data_path)
    arxiv_local = [
        fname.replace(".txt", "") for fname in fnames if fname.endswith(".txt")
    ]
    arxiv_done = db.get_arxiv_id_list(pu.db_params, "arxiv_qna")
    arxiv_pending = list(set(arxiv_local) - set(arxiv_done))
    print(f"Found {len(arxiv_pending)} papers pending.")

    with get_openai_callback() as cb:
        for arxiv_code in tqdm(arxiv_pending):
            ## Open doc and meta_data.
            doc_txt = pu.load_local(arxiv_code, data_path, False, "txt")
            doc_meta = pu.load_local(arxiv_code, meta_path, False, "json")
            authors_str = f"{doc_meta['authors'][0]['name']}, et al."
            year_str = doc_meta["published"][:4]
            doc_texts = text_splitter.split_text(doc_txt)
            doc_chunks = doc_texts[1:-1]

            print("\n", len(doc_chunks))
            print(f"(Authors: {authors_str}, Year: {year_str})")

            ## Run through Q&A pipeline.
            doc_qna_list = []

            for tid, text in tqdm(enumerate(doc_chunks), total=len(doc_chunks)):
                args = {
                    "authors": authors_str,
                    "year": year_str,
                    "arxiv_code": arxiv_code,
                    "text_chunk": text,
                }
                for _ in range(3):
                    try:
                        result_dict = chain(args)
                        break
                    except Exception as e:
                        print(e)
                        print("Retrying...")
                        continue
                qna_list = json.loads(result_dict["function"].json())["qna_pairs"]
                [q.update({"chunk_id": tid}) for q in qna_list]
                [q.update({"question_id": i}) for i, q in enumerate(qna_list)]
                doc_qna_list.extend(qna_list)

            ## Price logging.
            print(cb)

            ## Store QnA in DB.
            doc_qna_df = pd.DataFrame.from_dict(doc_qna_list)
            doc_qna_df["arxiv_code"] = arxiv_code
            doc_qna_df["chunk_id"] = doc_qna_df["chunk_id"].astype(int)
            doc_qna_df["question_id"] = doc_qna_df["question_id"].astype(int)
            doc_qna_df["question"] = doc_qna_df["question"].str.replace(
                "According to the LLM literature, ", ""
            )
            doc_qna_df["question"] = doc_qna_df["question"].apply(
                lambda x: x[0].upper() + x[1:] if x else x
            )
            doc_qna_df["answer"] = doc_qna_df["answer"].str.replace(
                "According to the LLM literature, ", ""
            )
            doc_qna_df["answer"] = doc_qna_df["answer"].apply(
                lambda x: x[0].upper() + x[1:] if x else x
            )
            db.upload_df_to_db(doc_qna_df, "arxiv_qna", pu.db_params)

            ## Store QnA in JSON.
            doc_qna_list_final = doc_qna_df.to_dict(orient="records")
            pu.store_local(doc_qna_list_final, arxiv_code, qna_path, relative=False)
            print(f"Stored {len(doc_qna_list_final)} Q&A pairs for {arxiv_code}.")

            ## Store document chunks in DB.
            doc_chunks_df = pd.DataFrame.from_dict(doc_chunks)
            doc_chunks_df["arxiv_code"] = arxiv_code
            doc_chunks_df["chunk_id"] = doc_chunks_df.index
            doc_chunks_df.columns = ["text", "arxiv_code", "chunk_id"]
            db.upload_df_to_db(doc_chunks_df, "arxiv_chunks", pu.db_params)

            ## Store document chunks in JSON.
            doc_chunks_list = doc_chunks_df.to_dict(orient="records")
            pu.store_local(doc_chunks_list, arxiv_code, chunk_path, relative=False)
            print(f"Stored {len(doc_chunks_list)} chunks for {arxiv_code}.")


if __name__ == "__main__":
    main()
