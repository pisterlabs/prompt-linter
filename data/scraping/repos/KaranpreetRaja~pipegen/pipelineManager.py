import weaviate
import json
import os
import requests
from dotenv import load_dotenv
import cohere
from cohere.custom_model_dataset import CsvDataset, InMemoryDataset, JsonlDataset


# load WEVIATE key and url and OpenAI key from .env file
load_dotenv()
weaviateKey = os.getenv('WEVIATE_KEY')
weaviateURL = os.getenv('WEVIATE_URL')
openaiKey = os.getenv('OPENAI_KEY')
cohereKey = os.getenv('COHERE_KEY')

class Pipeline:
    def __init__(self, request):
        self.request = request
        self.name = request['name']
        self.description = request['description']
        self.author = request['author']
        self.created = request['created']
        self.last_updated = request['last_updated']
        self.visibility_public = request['visibility_public']
        self.has_upload = request['has_upload']
        self.dynamic_upload = request['dynamic_upload'] if self.request['has_upload'] else False
        self.uploads = self.handle_uploads(request)
        self.model = self.handle_model(request)

        self.co = cohere.Client(cohereKey)

    def handle_uploads(self, request):
        # Logic to handle and store uploads
        uploads = []
        for upload in request['uploads']:
            current_upload = {
                'name': upload['name'],
                'type': upload['type'],
                'content': upload['content'],
                'reference': upload['reference']
            }
            uploads.append(current_upload)
        return uploads

    def handle_model(self, request):
        model = request['model']
        current_model = {
            'type': model['type'],
            'is_custom': model['is_custom'],
            'train_file_format': model['train_file_format'] if model['is_custom'] else 'N/A',
            'train_file': model['train_file'] if model['is_custom'] else 'N/A',
            'has_test': model['has_test'] if model['is_custom'] else False,
            'test_file': model['test_file'] if model['is_custom'] and model['has_test'] else 'N/A',
            'generation': model['generation']
        }
        return [current_model]

    def process_data(self, document_list):
        '''
        This function processes the data from each upload and adds it to a Weaviate vector database
        '''

        weaviate_client = weaviate.Client(
            url = weaviateURL,
            auth_client_secret=weaviate.AuthApiKey(api_key=weaviateKey),
            additional_headers = {
                "X-OpenAI-Api-Key": openaiKey,
            }
        )

        weaviate_client.schema.get()

        class_obj = {
            "class": "Document",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {},
                "generative-openai": {}
            }
        }
        weaviate_client.schema.create_class(class_obj)

        weaviate_client.batch.configure(batch_size=100)  # Configure batch
        with weaviate_client.batch as batch:  # Initialize a batch process
            for document in document_list:  # Batch import data
                batch.add_data_object(
                    data_object=document,
                    class_name="Document"
                )

        weaviate_client.batch.create()  # Execute batch process

        # save database to class
        self.database = weaviate_client

    

    def search_data(self, search_query):
        search_result = self.database.query.get('Document').with_near_text(search_query).do()
        documents = [result['chunk'] for result in search_result['data']['Get']['Document']]
        return documents


    def create_pipeline(self):
        # logic to create the model in the database
        # if custom model, call create_custom_model() first
        if self.is_custom:
            model = self.create_custom_model()
        else:
            # use the default model from cohere
            model = self.co.get_model(self.model['type'])

    def create_custom_model(self):
        
        if self.model['train_file_format'] == 'csv':
            dataset = CsvDataset(train_file=self.model['train_file'], delimiter=",")
        elif self.model['train_file_format'] == 'jsonl':
            dataset = JsonlDataset(train_file=self.model['train_file'])
        # Add other file formats if necessary
        
        finetune = self.co.create_custom_model(self.name, dataset=dataset, model_type=self.model['type'])
        return finetune


    def create_pipeline(self):
        if self.model['is_custom']:
            model_id = self.create_custom_model()
            model = self.co.get_model(model_id)
        else:
            # use the default model from cohere
            model = self.co.get_model(self.model['type'])

        # add RAG capabilities
        rag_model = self.add_rag_to_model(model)

        return rag_model


    def add_rag_to_model(self, model, prompt, documents):
        rag_model = self.co.chat(
            model_id=model,
            message=prompt,
            documents=documents,
            connectors=[{"id": "web-search"}]
        )
        return rag_model

    
    def call_rag_model(self, prompt):
        response = self.pipeline.chat(
            message=prompt,
            documents=self.database.get_documents(),
        )
        return response


    def export_as_json(self):
        # TODO: implement good logic to export the pipeline as a JSON file
        return self.request



def initialize_weaviate():
    '''
    This function processes the data from each upload and adds it to a Weaviate vector database
    '''

    client = weaviate.Client(
        url = weaviateURL,
        auth_client_secret=weaviate.AuthApiKey(api_key=weaviateKey),
        additional_headers = {
            "X-OpenAI-Api-Key": openaiKey,
        }
    )

    print(client.schema.get())

initialize_weaviate()