import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import openai
import dotenv,os
from transformers import GPT2TokenizerFast
import pickle
import time
from src.utils.rewrite_pages import rewrite
import sys
from pathlib import Path, PurePath
sys.path.append(PurePath(Path(__file__).parents[1]).as_posix())
from utils.logging.custom_logging import logger

# set api key
env = dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# enter the document name for which vector to be created
document_name = str(input('Enter PDF document name for which vector to be created(keep it short ex: pdp): '))

# pdf to text

pdfFileObj = open('../data/TVS Jupiter 125 - SMW.pdf', 'rb')
pdfReader = PyPDF2.PdfReader(pdfFileObj)
num_pages = len(pdfReader.pages)
data = []
logger.debug("wait while pages are being rewritten by completion API to remove noises")
for page in range(0, num_pages):
    pageObj = pdfReader.pages[page]
    page_text = pageObj.extract_text()
    data.append(page_text)
pdfFileObj.close()
data_rewrite = [rewrite(doc) for doc in data]
logger.info(f'Number of pages in the document is: {len(data)}')

# Split small chucks to so that LLMs can perform well
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
sources = None
for i, d in enumerate(data_rewrite):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": i}] * len(splits))

df = pd.DataFrame(metadatas)
df.insert(1, 'content', docs)
df.insert(1,'raw_index', df.index)
df = df.set_index(['raw_index',"source"])
logger.info(f'Number of rows in the document after chunk splits: {str(len(df))}')


# Tokenize

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") ##Todo: Use the logic provided by openai

content_token = [ count_tokens(text) for text in df.content.tolist()]
logger.info(f'Total number of tokens in document: {(str(sum(content_token)))}')
df.insert(1, 'tokens', content_token)


EMBEDDING_MODEL = "text-embedding-ada-002"


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
        model=model,
        input= text
    )
    return result["data"][0]["embedding"]


def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    logger.info(f'Embedding process is started')
    counter = 0
    embed_dict = {}
    page_count = 20 # For free-trail users, 20 requests per min are allowed
    for idx, r in df.iterrows():
        embed_dict[idx] = get_embedding(r.content)
        counter = counter + 1
        time.sleep(2)
        if counter == page_count:
            counter = 0
            logger.info(f'Embedding vector for {page_count} pages created.Waiting for 60 seconds before continuing')
            time.sleep(60) # Workaround for rate limit for a min
    logger.info(f'Embedding process is completed')
    return embed_dict


# compute embedding for the document
document_embeddings = compute_doc_embeddings(df)

# Save as pkl file
root_path = PurePath(Path(__file__).parents[1]).as_posix()
vector_path = os.path.join(root_path, 'vectorstores', f'{document_name}')
os.makedirs(vector_path, exist_ok=True)
# write docs.index and pkl file
df.to_pickle(os.path.join(vector_path,'df.pkl'))
df.to_csv(os.path.join(vector_path,'df.csv'))
with open(os.path.join(vector_path,"document_embeddings.pkl"), "wb") as f:
     pickle.dump(document_embeddings, f)
# end
# Todo: Update path in HTML so that new document can be recognized by UI
logger.info('Vectorization is successful')