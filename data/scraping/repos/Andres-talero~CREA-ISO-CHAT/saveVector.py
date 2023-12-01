import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.document_loaders import AzureBlobStorageContainerLoader
from langchain.text_splitter import CharacterTextSplitter
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    ScoringProfile,
    TextWeights,
)


def save_vector(text, url, company):
    vector_store_address = f"https://{os.environ.get('AZURE_COGNITIVE_SEARCH_SERVICE_NAME')}.search.windows.net"

    embeddings = OpenAIEmbeddings(deployment="embedding")

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=len(embeddings.embed_query("Text")),
            vector_search_configuration="default",
        ),
        SearchableField(
            name="metadata",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SearchableField(
            name="company",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
    ]

    index_name = os.environ.get('AZURE_COGNITIVE_SEARCH_INDEX_NAME')
    vector_store = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=os.environ.get("AZURE_COGNITIVE_SEARCH_API_KEY"),
        index_name=index_name,
        embedding_function=embeddings.embed_query,
        fields=fields,
    )

    text_splitter = CharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200, length_function=len)
    docs = text_splitter.create_documents(
        [text], metadatas=[{"company": company, "source": url}])
    print(docs)
    vector_store.add_documents(documents=docs)

    return "Data loaded into Azure Cognitive Search successfully"
