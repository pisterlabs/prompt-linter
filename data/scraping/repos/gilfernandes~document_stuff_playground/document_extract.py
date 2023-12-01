import sys
from pathlib import Path
from typing import List, Union
from langchain.docstore.document import Document
from langchain.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from dataclasses import dataclass
import pickle

from dotenv import load_dotenv

load_dotenv()

class Config(): 
    model = 'gpt-3.5-turbo-16k'
    # model = 'gpt-4'
    llm = ChatOpenAI(model=model, temperature=0)
    embeddings = OpenAIEmbeddings()
    chunk_size = 2000
    chroma_persist_directory = 'chroma_store'
    candidate_infos_cache = Path('candidate_infos_cache')
    if not candidate_infos_cache.exists():
        candidate_infos_cache.mkdir()

cfg = Config()

questions = [
    "What is the name of the job candidate?",
    "What are the specialities of this candidate?",
    "Please extract all hyperlinks.",
    "How many years of experience does this candidate have as a mobile developer?",
    "Which universities are mentioned in the CV?"
]

@dataclass
class CandidateInfo():
    """
    Contains the name of the candidate and the question / answer list.
    """
    candidate_file: str
    questions: list[(str, str)]


def process_document(doc_path) -> Chroma:
    """
    Processes the document by loading the text from the document. 
    There are two supported formats: pdf and docx. Then it splits 
    the text in large chunks from which then embeddings are extracted.
    :param doc_path a path with documents or a string representing that path.
    :return a Chroma wrapper around the embeddings.
    """
    if not isinstance(doc_path, Path):
        doc_path = Path(doc_path)
    if not doc_path.exists():
        print(f"The document ({doc_path}) does not exist. Please check")
    else:
        print(f"Processing {doc_path}")
        loader = (PDFPlumberLoader(str(doc_path)) if doc_path.suffix == ".pdf"
                  else Docx2txtLoader(str(doc_path)))
        doc_list: List[Document] = loader.load()
        print(f"Extracted documents: {len(doc_list)}")
        for i, doc in enumerate(doc_list):
            i += 1
            if len(doc.page_content) == 0:
                print(f"Document has empty page: {i}")
            else:
                print(f"Page {i} length: {len(doc.page_content)}")
        text_splitter = CharacterTextSplitter(chunk_size=cfg.chunk_size, chunk_overlap=0)
        texts = text_splitter.split_documents(doc_list)

        return extract_embeddings(texts, doc_path)
    

def extract_embeddings(texts: List[Document], doc_path: Path) -> Chroma:
    """
    Either saves the Chroma embeddings locally or reads them from disk, in case they exist.
    :return a Chroma wrapper around the embeddings.
    """
    embedding_dir = f"{cfg.chroma_persist_directory}/{doc_path.stem}"
    if Path(embedding_dir).exists():
        return Chroma(persist_directory=embedding_dir, embedding_function=cfg.embeddings)
    try:
        docsearch = Chroma.from_documents(texts, cfg.embeddings, persist_directory=embedding_dir)
        docsearch.persist()
    except Exception as e:
        print(f"Failed to process {doc_path}: {str(e)}")
        return None
    return docsearch


def read_saved_candidate_infos(file_key: str) -> Union[None, CandidateInfo]:
    """
    Reads a pickle file with the questions and answers about a candidate.
    :param file_key The key - file name used to retrieve the pickle file.
    :return either nothing or a set of questions and answers.
    """
    cached_file = cfg.candidate_infos_cache/file_key
    try:
        if cached_file.exists():
            with open(cached_file, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Could not process {file_key}")
    return None


def write_candidate_infos(file_key, candidate_info):
    """
    Writes a pickle file with the questions and answers about a candidate.
    :param file_key The key - file name used to retrieve the pickle file.
    :candidate_info The information about a candidate which will be pickled.
    """
    cached_file = cfg.candidate_infos_cache/file_key
    with open(cached_file, "wb") as f:
        pickle.dump(candidate_info, f)


def extract_candidate_infos(doc_folder: Path) -> List[CandidateInfo]:
    """
    Extracts the questions and answers from each pdf or docx file in `doc_folder` 
    and saves these in a list. First it loops through the files, extracts their content
    as embeddings and caches these and then interacts with ChatGPT. The answers are then 
    saves in a data structure and cached. If the naswers are alwready available for a candidate
    they are read from a pickled file.
    :param doc_folder The folder with the candidate documents.
    :return the list with candidate question' and answers.
    """
    if not doc_folder.exists():
        print(f"Candidate folder {doc_folder} does not exist!")
        return []
    candidate_list: list[CandidateInfo] = []
    extensions: list[str] = ['**/*.pdf', '**/*.docx']
    for extension in extensions:
        for doc in doc_folder.rglob(extension):
            file_key = doc.stem
            cached_candidate_info = read_saved_candidate_infos(file_key)
            if cached_candidate_info is None:
                docsearch = process_document(doc)
                print(f"Processed {doc}")
                if docsearch is not None:
                    qa = RetrievalQA.from_chain_type(llm=cfg.llm, chain_type="stuff", retriever=docsearch.as_retriever())
                    question_list = []
                    for question in questions:
                        question_list.append((question, qa.run(question)))
                    candidate_info = CandidateInfo(candidate_file=file_key, questions=question_list)
                    write_candidate_infos(file_key, candidate_info)
                    candidate_list.append(candidate_info)
                else:
                    print(f"Could not retrieve content from {doc}")
            else:
                candidate_list.append(cached_candidate_info)
    return candidate_list


def render_candidate_infos(candidate_infos: list[CandidateInfo]) -> str:
    """
    Receives a list of candidate question and answers and converts them to HTML.
    :param candidate_infos The list of candidate question and answers
    :return an HTML string.
    """
    html = ""
    for candidate_info in candidate_infos:
        qa_html = ""
        for question, answer in candidate_info.questions:
            qa_html += f"""
<h5 class="card-title">{question}</h5>
<p class="card-text"><pre style="background-color: #f6f8fa; padding: 1em">{answer}</pre></p>
"""
        html += f"""
<div class="card">
  <div class="card-header" style="cursor: pointer">
    {candidate_info.candidate_file}
  </div>
  <div class="card-body mb-3">
    {qa_html}
  </div>
</div>
"""
    return html