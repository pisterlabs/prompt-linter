import os

# from dotenv import load_dotenv
from chromadb.config import Settings

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/data"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8

# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlxs": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# Default Instructor Model
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
# You can also choose a smaller model, don't forget to change HuggingFaceInstructEmbeddings
# to HuggingFaceEmbeddings in both ingest.py and run_localGPT.py
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

DEVICE_TYPE = "cuda"

# for HF models
# MODEL_ID = "TheBloke/vicuna-7B-1.1-HF"
# MODEL_ID = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
MODEL_ID = "TheBloke/guanaco-7B-HF"
# MODEL_ID = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM.
# Using STransformers alongside will 100% create OOM on 24GB cards.

# for GPTQ (quantized) models
# MODEL_ID = "TheBloke/Nous-Hermes-13B-GPTQ"
# MODEL_BASENAME = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
# MODEL_ID = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
# MODEL_BASENAME = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors"
# Requires ~21GB VRAM. Using STransformers alongside can potentially create OOM on 24GB cards.
# MODEL_ID = "TheBloke/wizardLM-7B-GPTQ"
# MODEL_BASENAME = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"
MODEL_ID = "TheBloke/WizardLM-7B-uncensored-GPTQ"
MODEL_BASENAME = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
