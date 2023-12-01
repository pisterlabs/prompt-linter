from fpdf import FPDF
import textwrap
from dataclasses import asdict, dataclass
from typing import List
from langchain import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
import re
import PyPDF2
from io import BytesIO
import difflib

# from langchain.evaluation import load_evaluator, EmbeddingDistance
# evaluator = load_evaluator(
#     "pairwise_embedding_distance", distance_metric=EmbeddingDistance.COSINE
# )

_CHAT_MODEL_NAME = "gpt-3.5-turbo"
_CHUNK_SIZE = 512
_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

LANGCHAIN_SPLITTER = TokenTextSplitter(
    encoding_name=_CHAT_MODEL_NAME,
    model_name=_CHAT_MODEL_NAME,
    chunk_size=_CHUNK_SIZE,
    chunk_overlap=0,
)


@dataclass
class MetaData:
    source: str


def create_docs(text_chunks: str, url: str = None) -> List[Document]:
    documents = []
    # if doc_id is None:
    #     doc_id = str(uuid4().hex)  # generate random doc id

    for chunk_id, chunk_text in enumerate(text_chunks):
        # print(chunk_text)
        metadata = MetaData(
            source=chunk_id,
        )
        documents.append(Document(page_content=chunk_text,
                         metadata=asdict(metadata)))
    return documents


def get_documents(text: str):
    text_chunk = LANGCHAIN_SPLITTER.split_text(text)
    documents = create_docs(text_chunk)
    return documents


def create_pdf_from_text(text):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(True, margin=10)  # Margin adjusted for page breaks
    pdf.add_page()
    pdf.set_font(family='Arial', size=10)  # Adjust font and size
    splitted = text.split('\n')

    for line in splitted:
        lines = textwrap.wrap(line, width=115)  # Adjust width

        if len(lines) == 0:
            pdf.ln()

        for wrap in lines:
            pdf.cell(0, 7, wrap, ln=1)  # Adjust line height

    byte_string = pdf.output(dest="S")
    buffer = BytesIO(byte_string.encode("latin1"))
    buffer.seek(0)
    return buffer


def text_from_message(messages):
    contents = []
    for message in messages:
        if "assistant" in message.get("role", ""):
            contents.append(message.get("content", "").replace("\n\n", "\n"))
            contents.append("\n\n")
    texts = " ".join(contents)
    return texts


def generate_pdf(messages):
    texts = text_from_message(messages)
    return create_pdf_from_text(texts)


def generate_diff_pdf(messages, pdf_text):
    d = difflib.Differ()
    texts = text_from_message(messages)
    diff = list(d.compare(pdf_text.splitlines(), texts.splitlines()))
    diff_text = "\n".join(diff)
    return create_pdf_from_text(diff_text)


def read_pdf_with_pypdf2(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    text = ""
    for page_number in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_number]
        text += page.extract_text()

    return text


def get_vectorstore(data):
    _EMBEDDING_MODEL = OpenAIEmbeddings(
        model=_EMBEDDING_MODEL_NAME, deployment=_EMBEDDING_MODEL_NAME
    )
    # search_index = FAISS.load_local(data,embeddings=_EMBEDDING_MODEL)
    search_index = FAISS.from_documents(data, _EMBEDDING_MODEL)
    return search_index


def html_parser(response):
    # <p><img src="https://files.readme.io/643df24-Column_Names.jpg" width="780" height="400" alt="Column Names"/><br/>
    responses = re.split(r'<<<|>>>', response)
    if len(responses) == 1:
        return f'<p> {responses[0]} </p> <br>'
    parsed_response = ""
    images = []  # To remove the images that are repeated
    for res in responses:
        if "drive.google.com" in res or "cdn.embedly.com" in res or "support.google.com" in res:
            continue
        if "http" in res:
            if res in images:
                continue
            parsed_response += f'<p><img src="{res.strip()}" width="640" height="300"/></p> <br>'
            images.append(res)
            continue
        parsed_response += f'<p> {res} </p> <br>'

    return parsed_response


def jaccard_similarity(set1: set, set2: set) -> float:
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def jaccard_similarity_list(resp: str, extracted_chunk: List[str]) -> List[float]:
    sim_score = []
    for text in extracted_chunk:
        resp_ = set(
            word for word in resp.split()
        )
        text_ = set(
            word for word in text.split()
        )
        similarity = jaccard_similarity(resp_, text_)
        sim_score.append(similarity)
    return sim_score

# def get_similarity_score(doc_text, pred_text):
#     score =evaluator.evaluate_string_pairs(
#         prediction=doc_text, prediction_b=pred_text
#     )
#     return score
