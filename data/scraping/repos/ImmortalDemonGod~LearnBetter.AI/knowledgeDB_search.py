import argparse
import pyperclip
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
# Import SyncGoTrueClient directly
from gotrue import SyncGoTrueClient
from supabase import create_client
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
from dotenv import dotenv_values

#logging.basicConfig(level='OFF')
#logging.getLogger('transformers').setLevel(logging.ERROR)
#logging.basicConfig(level='CRITICAL') 

logging.getLogger('sentence_transformers').setLevel(logging.CRITICAL) 
logging.getLogger('supabase').setLevel(logging.CRITICAL) 
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

# Load the .env file
load_dotenv()

config = dotenv_values(".env")  # load shared development variables

SUPABASE_URL = config['SUPABASE_URL']
SUPABASE_SERVICE_KEY = config['SUPABASE_SERVICE_KEY']
BRAIN_IDS = {
    "default": "f04f26fe-ed6b-428e-8d2e-1e5917814b1c",
    "programming": "fbcf7baf-9a85-4941-b9cf-c2f94825aedd",
    "english_writing": "c7355034-3f92-4504-baec-a54ee3da1958",
    "youtube_history": "f613ec2b-0516-44af-9238-3baae3c3bdbc"
}
parser = argparse.ArgumentParser(description='KnowledgeDatabase Brain ID')
parser.add_argument('--brain_id', type=str, help='Brain ID to use for searching', default='default')
args = parser.parse_args()

BRAIN_ID = BRAIN_IDS.get(args.brain_id, BRAIN_IDS['default'])
TABLE_NAME = config['TABLE_NAME']
model_name = config.get('MODEL_NAME')  # provide a default value if not specified in .env
LOCAL_EMBEDDINGS = HuggingFaceEmbeddings(model_name=model_name)
print(SUPABASE_URL)
print(SUPABASE_SERVICE_KEY)

def parse_arguments():
    parser = argparse.ArgumentParser(description="KnowledgeDatabase search command for Autogpt. Example usage: python knowledgeDB_search.py 'search query1' 'search query2' --num_results 10")
    parser.add_argument("query", nargs='+', type=str, help="The search query")
    parser.add_argument("--num_results", type=int, default=5, help="The number of results to return")
    return parser.parse_args()


def _knowledgedb_search(query: str, num_results: int = 5) -> str | List[Document]:
    brain_id = BRAIN_ID
    table = "match_vectors"
    k = num_results
    threshold = 0.5

    query_embedding = LOCAL_EMBEDDINGS.embed_query(query)

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    result = supabase.rpc(
        table,
        {
            "query_embedding": query_embedding,
            "match_count": k,
            "p_brain_id": str(brain_id)
        },
    ).execute()

    match_result = [
        (
            Document(
                metadata=search.get("metadata", {}),
                page_content=search.get("content", ""),
            ),
            search.get("similarity", 0.0),
        )
        for search in result.data
        if search.get("content")
    ]

    for doc, _ in match_result:
        doc.page_content = doc.page_content + f"\nSOURCES: {doc.metadata}"

    documents = [doc for doc, _ in match_result]
    # print(type(documents))
    return documents


out_text = ""

def save_output_to_file(query, output, filename):
    global out_text

    with open(filename, 'a') as f:
        out_text += f'Query:\n{query}\nOutput:\n'
        f.write(f'Query:\n{query}\nOutput:\n')
        for item in output:
            f.write(str(item) + '\n')
            out_text += str(item) + '\n'
        f.write('\n')
        out_text += '\n'

if __name__ == "__main__":
    args = parse_arguments()

    base_filename = "output_files/" + input("Please enter the base filename: ")
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Append timestamp to filename
    new_filename = f"{base_filename}_{timestamp}"
    for query in args.query:
        output = _knowledgedb_search(query, args.num_results)
        save_output_to_file(query, output, new_filename)
    pyperclip.copy(out_text)
    print("Success")
