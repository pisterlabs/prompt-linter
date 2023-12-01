from omegaconf import DictConfig
import hydra
from src.entities.vector_db_params import VectorDBParams, read_vector_db_params
import logging
import os
from collections import defaultdict
from tqdm import tqdm
import re
import chromadb
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch import Tensor
import cohere
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import fitz
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]


def get_page_number(filename):
    match = re.match(r"(\d+)_", filename)
    return int(match.groups()[0]) if match else None


def read_file_content(filename):
    with open(filename, 'r') as file:
        return file.read()


def process_page(files, **kwargs):
    combined_text = '\n'.join([read_file_content(file) for file in files])
    tensor_list = tokenize_and_chunk(combined_text, **kwargs)
    return tensor_list


def get_cohere_embeddings(texts, source_docs, model_name, co):
    doc_emb = co.embed([source_doc + '\n' + text for text, source_doc in zip(texts, source_docs)], input_type="search_document", model=model_name).embeddings
    return doc_emb


def tokenize_and_chunk(text: str, **kwargs):
    # Tokenize the input text
    text_with_doc_name = kwargs['prefix'] + kwargs['doc_name'] + '\n' + text
    tokens = kwargs['tokenizer'].encode_plus(
        text_with_doc_name,
        add_special_tokens=True,
        max_length=kwargs['max_length'],
        truncation=False,
        return_tensors='pt'
    )

    if tokens['input_ids'].size(1) > kwargs['max_length']:
        # Find the position to split the text, avoiding splitting inside a word
        split_position = text.rfind('\n', 0, len(text) // 2)
        if split_position <= 0 or split_position >= len(text) // 2:
            split_position = text.rfind(' ', 0, len(text) // 2)

        # Split the text into two halves and recursively call this function
        first_half = tokenize_and_chunk(text[:split_position], **kwargs)
        second_half = tokenize_and_chunk(text[split_position:], **kwargs)

        return first_half + second_half
    else:
        return [(tokens, text)]


def get_embeddings(batch_dict, model):
    outputs = model(**batch_dict)

    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()


def create_text_embeddings(params: VectorDBParams):
    cohere_model = False
    if params.embedding_model != 'Cohere/Cohere-embed-multilingual-v3.0':
        model = AutoModel.from_pretrained(params.embedding_model)
    else:
        load_dotenv()
        co = cohere.Client(os.getenv('COHERE_API_KEY'))
        cohere_model = True
        cohere_requests = {
            'texts': [],
            'source_docs': [],
            'page_numbers': []
        }
    tokenizer = AutoTokenizer.from_pretrained(params.embedding_model)
    client = chromadb.PersistentClient(path=params.db_path)
    try:
        client.delete_collection(name=params.collection_name)
    except ValueError:
        print(f"Collection {params.collection_name} does not exist.")
    collection = client.create_collection(name=params.collection_name,
                                          metadata={'hnsw:space': params.distance_metric},
                                          )

    count = 0
    for source_doc in tqdm(os.listdir(params.input_path)):
        files_by_page = defaultdict(list)
        for filename in os.listdir(f'{params.input_path}/{source_doc}'):
            page_number = get_page_number(filename)
            if page_number is not None:
                files_by_page[page_number].append(filename)
        for page_number in sorted(files_by_page):
            files_by_page[page_number].sort(key=natural_keys)
            tensor_pages = process_page([f'{params.input_path}/{source_doc}/{file}'
                                        for file in files_by_page[page_number]
                                        if params.text_embeddings.use_tables and file.endswith('.txt') or not params.text_embeddings.use_tables and file.endswith('text.txt')],
                                        doc_name=source_doc,
                                        tokenizer=tokenizer,
                                        max_length=params.text_embeddings.max_tokens,
                                        prefix=params.text_embeddings.prefix
                                        )
            for tensor, text in tensor_pages:
                if not cohere_model:
                    embeddings = get_embeddings(tensor, model)
                    collection.add(
                        embeddings=embeddings,
                        documents=[text],
                        metadatas=[{"page": page_number, "document_name": source_doc}],
                        ids=[str(count)],
                    )
                    count += 1
                else:
                    cohere_requests['texts'].append(text)
                    cohere_requests['source_docs'].append(source_doc)
                    cohere_requests['page_numbers'].append(page_number)
    if cohere_model:
        embeddings = get_cohere_embeddings(cohere_requests['texts'], cohere_requests['source_docs'], 'embed-multilingual-v3.0', co)
        collection.add(
            embeddings=embeddings,
            documents=cohere_requests['texts'],
            metadatas=[{"page": page_number, "document_name": source_doc} for page_number, source_doc in zip(cohere_requests['page_numbers'], cohere_requests['source_docs'])],
            ids=[str(count) for count in range(len(cohere_requests['texts']))],
        )


def create_clip_embeddings(params: VectorDBParams):
    count = 0
    img_model = SentenceTransformer(params.embedding_model)
    mat = fitz.Matrix(1, 1)
    client = chromadb.PersistentClient(path=params.db_path)
    try:
        client.delete_collection(name=params.collection_name)
    except ValueError:
        print(f"Collection {params.collection_name} does not exist.")
    collection = client.create_collection(name=params.collection_name,
                                          metadata={'hnsw:space': params.distance_metric},
                                          )
    for source_doc in tqdm(os.listdir(params.input_path)):
        if source_doc.endswith('.pdf'):
            doc = fitz.open(f'{params.input_path}/{source_doc}')
            doc_name = source_doc.replace('.pdf', '')
            os.makedirs(f'{params.clip_embeddings.images_folder}/{doc_name}', exist_ok=True)
            for page_id, page in enumerate(doc):
                pix = page.get_pixmap(matrix=mat)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                draw = ImageDraw.Draw(image)

                font = ImageFont.truetype("Helvetica", params.clip_embeddings.font_size)
                draw.text(tuple(params.clip_embeddings.x_y_docname_loc), doc_name, font=font, fill=(0, 0, 0))
                image.save(f'{params.clip_embeddings.images_folder}/{doc_name}/{page_id}.png')
                img_embeddings = img_model.encode([image], normalize_embeddings=True)
                collection.add(
                    embeddings=img_embeddings.tolist(),
                    metadatas=[{"page": page_id, "document_name": doc_name}],
                    uris=[f'{params.clip_embeddings.images_folder}/{doc_name}/{page_id}.png'],
                    ids=[str(count)],
                )
                count += 1


def create_vector_db(params: VectorDBParams):
    logger.info(f"start creating vector db with params {params}")
    os.makedirs(params.db_path, exist_ok=True)
    if params.embeddings == 'text':
        create_text_embeddings(params)
    elif params.embeddings == 'clip':
        create_clip_embeddings(params)


@hydra.main(version_base=None, config_path="../configs", config_name="vector_db_config")
def prepare_vector_db_pipeline_command(cfg: DictConfig):
    params = read_vector_db_params(cfg)
    create_vector_db(params)


if __name__ == "__main__":
    prepare_vector_db_pipeline_command()
