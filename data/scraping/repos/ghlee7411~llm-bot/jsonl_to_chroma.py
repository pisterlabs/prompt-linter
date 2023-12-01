from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import JSONLoader
from pprint import pprint
from tqdm import tqdm
import json

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv(verbose=True, override=True)
del load_dotenv


def main():
    # Document Loader
    jsonl_path = 'recipe_items.jsonl'
    recipe_items = list()
    batch_size = 50
    with open(jsonl_path, 'r') as f:
        batch = list()

        # Document Transformer
        for line in tqdm(f):
            json_data = json.loads(line)
            
            # remove unnecessary keys
            if '_id' in json_data.keys():
                del json_data['_id']
            if 'source' in json_data.keys():
                del json_data['source']
            if 'url' in json_data.keys():
                del json_data['url']
            if 'image' in json_data.keys():
                del json_data['image']
            if 'ts' in json_data.keys():
                del json_data['ts']

            # convert ingredients from str to list
            if 'ingredients' in json_data.keys():
                ingredients = json_data['ingredients'] # str
                ingredients = ingredients.split('\n') # list
                json_data['ingredients'] = ingredients

            if len(batch) < batch_size:
                batch.append(json.dumps(json_data))
            else:
                recipe_items.append(batch)
                batch = list()
        
        if len(batch) > 0:
            recipe_items.append(batch)
            batch = list()

    # Initialize a Text Embedding Model
    embeddings = OpenAIEmbeddings()

    # Warning!! This will take a long time. Cost expensive tasks.

    # Initialize Chroma from persisted directory (VectorStore)
    db = Chroma(persist_directory='recipe_items_chroma', embedding_function=embeddings)

    for batch in tqdm(recipe_items, total=len(recipe_items), desc='recipe_items to db'):
        # print(recipe_item)
        db.add_texts(batch)


if __name__ == "__main__":
    main()