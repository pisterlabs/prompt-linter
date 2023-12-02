import os
import gradio as gr
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai  # for calling the OpenAI API
import os

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
SYSTEM_TEMPLATE = os.getenv('SYSTEM_TEMPLATE')
collection_name = os.getenv('MILVUS_COLLECTION_NAME')

#'''
# Search the database based on input text
def embed_search(data):
    # Create a SentenceTransformer model
    transformer = SentenceTransformer(os.getenv('EMBEDDING_MODEL'))
    embeds = transformer.encode(data)
    return [x for x in embeds]

def data_querying(input_text):
    print('Searching in vector DB ...')
    search_terms = [input_text]
    search_data = embed_search(input_text)
    # Connect to Milvus Database
    connections.connect(host=os.getenv('MILVUS_HOST'), port=os.getenv('MILVUS_PORT'), secure=False)
    collection = Collection(collection_name)
    response = collection.search(
        data=[search_data],  # Embedded search value
        anns_field="embedded_vectors",  # Search across embeddings
        param={},
        limit = 3,  # Limit to top_k results per search
        output_fields=['chunked_text', 'file_path']  # Include required field in result
    )
    
    prompt_text = ''
    for _, hits in enumerate(response):
        for hit in hits:
            prompt_text += hit.entity.get('chunked_text') + '\n\nSOURCE: ' + hit.entity.get('file_path') + '\n\n'

    query = SYSTEM_TEMPLATE.format(context=prompt_text, question=search_terms[0])
    output = openai.ChatCompletion.create(
        messages=[
            {'role': 'system', 'content': 'You answer questions related to financial documents'},
            {'role': 'user', 'content': query},
        ],
        model=os.getenv('GPT_MODEL'),
        temperature=0,
    )
    return output['choices'][0]['message']['content']


iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=7, label="Enter your question", ),
                     outputs="text",
                     title="Financial Knowledge Base",
                     description="Ask a question about the NASDAQ data and get a response",
                     ).queue()

iface.launch(share=False)