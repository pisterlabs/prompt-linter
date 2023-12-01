from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Milvus
import spacy

# important commands
# bin/kafka-console-consumer.sh --topic quickstart-events --from-beginning --bootstrap-server localhost:9092

def get_pdf_text(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def preprocess_text(text):
    text = text.lower()
    return text
path = "en_core_web_sm-3.0.0/en_core_web_sm/en_core_web_sm-3.0.0"
def coreferencing(text):
    # nlp = spacy.load("en_coreference_web_trf")
    nlp = spacy.load(path)
    doc = nlp (text)
    offset = 0
    reindex = []
    for chain in doc.spans:
        for idx, span in enumerate(doc.spans[chain]):        
            if idx > 0:
                reindex.append([span.start_char, span.end_char, doc.spans[chain][0].text])

    for span in sorted(reindex, key=lambda x:x[0]):    
        text = text[0:span[0] + offset] + span[2] + text[span[1] + offset:]
        offset += len(span[2]) - (span[1] - span[0])    
    
    return text

def create_embeddings(text):
    model = INSTRUCTOR('hkunlp/instructor-base')
    instruction = "Represent the text:"
    embeddings = model.encode([[instruction , sentence]])
    return embeddings
