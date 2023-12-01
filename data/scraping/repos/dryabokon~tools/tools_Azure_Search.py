#https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/search/azure-search-documents/samples/sample_vector_search.py
#----------------------------------------------------------------------------------------------------------------------
import openai
import pandas as pd
import yaml
#----------------------------------------------------------------------------------------------------------------------
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex,SearchField,SearchFieldDataType,SimpleField,SearchableField,VectorSearch,VectorSearchAlgorithmConfiguration#,HnswVectorSearchAlgorithmConfiguration
from openai import AzureOpenAI
#----------------------------------------------------------------------------------------------------------------------
#from azure.ai.textanalytics import TextAnalyticsClient
#----------------------------------------------------------------------------------------------------------------------
class Client_Search(object):
    def __init__(self, filename_config,index_name=None,filename_config_emb_model=None):
        if filename_config is None:
            return
        with open(filename_config, 'r') as config_file:
            self.config_search = yaml.safe_load(config_file)
            if not 'azure' in self.config_search.keys():
                return

            self.search_index_client = SearchIndexClient(self.config_search['azure']['azure_search_endpoint'], AzureKeyCredential(self.config_search['azure']['azure_search_key']))

        self.index_name = index_name if index_name is not None else self.config_search['azure']['index_name']
        self.search_client = self.get_search_client(self.index_name)

        if filename_config_emb_model is not None:
            with open(filename_config_emb_model, 'r') as config_file:
                self.config_emb = yaml.safe_load(config_file)
                openai.api_type = "azure"
                openai.api_version = self.config_emb['azure']['openai_api_version']
                openai.api_base = self.config_emb['azure']['openai_api_base']
                openai.api_key = self.config_emb['azure']['openai_api_key']
                self.embedding = AzureOpenAI(
                    api_key=self.config_emb['azure']['openai_api_key'],
                    api_version=self.config_emb['azure']['openai_api_version'],
                    azure_endpoint=self.config_emb['azure']['openai_api_base']
                ).embeddings

                self.model_deployment_name = self.config_emb['azure']['deployment_name']

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_search_client(self,index_name):
        return SearchClient(self.config_search['azure']['azure_search_endpoint'], index_name,AzureKeyCredential(self.config_search['azure']['azure_search_key']))
#----------------------------------------------------------------------------------------------------------------------
    def get_NER_client(self):
        return TextAnalyticsClient(endpoint=self.config_search['azure']['azure_search_endpoint'], credential=AzureKeyCredential(self.config_search['azure']['azure_search_key']))
# ----------------------------------------------------------------------------------------------------------------------
    def create_fields(self, docs, field_embedding):
        df = pd.DataFrame(docs[:1])

        dct_typ = {'object': SearchFieldDataType.String, 'int32': 'int', 'int64': 'int'}
        fields = []
        for r in range(df.shape[1]):
            name = df.columns[r]
            type = df.dtypes.iloc[r]
            if r==0:
                field = SimpleField(name=name, type=SearchFieldDataType.String, key=True)
            elif name==field_embedding:
                field = SearchField(name=name, type=SearchFieldDataType.Collection(SearchFieldDataType.Single),searchable=True, vector_search_dimensions=len(self.get_embedding("Text")),vector_search_configuration="default")
            else:
                field = SearchableField(name=name, type=SearchFieldDataType.String)
            fields.append(field)

        # fields = [SimpleField(name="hotelId", type=SearchFieldDataType.String, key=True),
        #             SearchField(name="descriptionVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),searchable=True, vector_search_dimensions=len(self.get_embedding("Text")),vector_search_configuration="default"),
        #             SearchableField(name="hotelName", type=SearchFieldDataType.String, sortable=True, filterable=True),
        #             SearchableField(name="description", type=SearchFieldDataType.String),
        #             SearchableField(name="category", type=SearchFieldDataType.String, sortable=True, filterable=True,facetable=True)]

        return fields
# ----------------------------------------------------------------------------------------------------------------------
    def tokenize_documents(self, dct_records, field_source, field_embedding):
        for d in dct_records:
            d[field_embedding]=self.get_embedding(d[field_source])

        return dct_records
# ----------------------------------------------------------------------------------------------------------------------
    def create_search_index(self,index_name,fields):
        #vector_search = VectorSearch(algorithm_configurations=[HnswVectorSearchAlgorithmConfiguration(name="default", kind="hnsw")])
        vector_search = VectorSearch(algorithm_configurations=[VectorSearchAlgorithmConfiguration(name="default", kind="hnsw")])
        return SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
# ----------------------------------------------------------------------------------------------------------------------
    def init_vector_store(self):
        #
        # self.vector_store = AzureSearch(
        #     azure_search_endpoint=self.config_search['azure']['azure_search_endpoint'],
        #     azure_search_key=self.config_search['azure']['openai_api_key'],
        #     index_name=self.index_name,
        #     embedding_function=self.embedding.embed_query,
        #     fields=fields)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_embedding(self,text):
        #embedding = self.embedding.create(input=text, deployment_id=self.model_deployment_name)["data"][0]["embedding"]
        embedding = self.embedding.create(input=text,model=self.model_deployment_name).data[0].embedding
        return embedding
# ----------------------------------------------------------------------------------------------------------------------
    def get_indices(self):
        res = [x for x in self.search_index_client.list_index_names()]
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def get_document(self,key):
        result = self.search_client.get_document(key=key)
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def upload_documents(self,docs):
        if not isinstance(docs,list):
            docs = [docs]
        result = self.search_client.upload_documents(documents=docs)
        return result[0].succeeded
# ----------------------------------------------------------------------------------------------------------------------
    def delete_document(self,dict_doc):
        self.search_client.delete_documents(documents=[dict_doc])
        return
# ----------------------------------------------------------------------------------------------------------------------
    def search_document(self, query,select=None):
        results = self.search_client.search(search_text=query,select=select)
        df = pd.DataFrame([r for r  in results])
        df = df.iloc[:,[c.find('@')<0 for c in df.columns]]
        if not select is None and df.shape[0]>0:
            df = df[select]

        return df
# ----------------------------------------------------------------------------------------------------------------------
    def search_document_hybrid(self, query,field_embedding,select=None):
        vector = Vector(value=self.get_embedding(query), k=3, fields=field_embedding)
        results = self.search_client.search(search_text=query,vectors=[vector],select=select)
        df = pd.DataFrame([r for r in results])
        df = df.iloc[:, [c.find('@') < 0 for c in df.columns]]
        if not select is None:
            df = df[select]
        return df
# ----------------------------------------------------------------------------------------------------------------------
    def search_texts(self,query,select=None):
        results = self.search_client.search(search_text=query,select=select)
        list_of_dict = [r for r in results]
        if isinstance(select,list):
            texts = [';'.join([x+':'+str(r[x]) for x in select]) for r in list_of_dict]
        else:
            texts = [r[select] for r in list_of_dict]
        return texts
# ----------------------------------------------------------------------------------------------------------------------
    def search_texts_hybrid(self,query,field,select=None,limit=4):

        #vector = Vector(value=self.get_embedding(query), fields=field)
        # results = self.search_client.search(search_text=query, vectors=[vector], select=select,top=limit)
        # results = self.search_client.search(search_text=None, vector=[vector], select=select)

        results = self.search_client.search(search_text=query, select=select,top=limit)
        list_of_dict = [r for r in results]
        texts = [r[select] for r in list_of_dict]
        return texts
# ----------------------------------------------------------------------------------------------------------------------