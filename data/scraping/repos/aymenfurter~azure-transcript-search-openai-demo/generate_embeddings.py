import os
import sys
import re
import time
import webvtt
from collections import defaultdict
from azure.identity import AzureDeveloperCliCredential
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *
from azure.search.documents import SearchClient
from azure.ai.formrecognizer import DocumentAnalysisClient
import openai
from pytube import YouTube


def initialize_search_index(acs_key, acs_instance):
    # search_cred should be TokenCredential
    auth_credentials = AzureKeyCredential(acs_key)

    search_client = SearchIndexClient(endpoint=f"https://{acs_instance}.search.windows.net/",
                                      credential=auth_credentials)
    if "embeddings" not in search_client.list_index_names():
        index_structure = SearchIndex(
            name="embeddings",
            fields=[
                SimpleField(name="Id", type="Edm.String", key=True),
                SearchableField(name="Text", type="Edm.String", analyzer_name="en.microsoft"),
                SearchableField(name="Description", type="Edm.String", analyzer_name="en.microsoft"),
                SearchableField(name="AdditionalMetadata", type="Edm.String", analyzer_name="en.microsoft"),
                SearchableField(name="ExternalSourceName", type="Edm.String", analyzer_name="en.microsoft"),
                SimpleField(name="CreatedAt", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
                SearchableField(name="ChannelName", type="Edm.String", analyzer_name="en.microsoft"),
                SearchableField(name="VideoName", type="Edm.String", analyzer_name="en.microsoft"),
                SearchField(name="Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
                            hidden=False, searchable=True, filterable=False, sortable=False, facetable=False,
                            dimensions=1536, vector_search_configuration="default"), 
            ],
            semantic_settings=SemanticSettings(
                configurations=[SemanticConfiguration(
                    name='standard',
                    prioritized_fields=PrioritizedFields(
                        title_field=None, prioritized_content_fields=[SemanticField(field_name='Text')]))]),
                vector_search=VectorSearch(
                    algorithm_configurations=[
                        VectorSearchAlgorithmConfiguration(
                            name="default",
                            kind="hnsw",
                            hnsw_parameters=HnswParameters(metric="cosine") 
                        )
                    ]
                )        
            )
        print(f"Initializing search index")
        search_client.create_index(index_structure)
    else:
        print(f"Search index already exists")


def create_embeddings(video_id):

    url = f'https://www.youtube.com/watch?v={video_id}'
    yt = YouTube(url)

    video_name = yt.title
    channel_name = yt.author
    publish_date = yt.publish_date



    file = 'data/' + video_id+'.en.vtt'
    captions = webvtt.read(file)

    result = defaultdict(str)
    last_words = []
    for caption in captions:
        start_time = caption.start
        hh, mm, ss_ms = start_time.split(':')
        ss, ms = ss_ms.split('.')
        key = int(hh) * 60 + int(mm)

        current_words = caption.text.strip().split()
        overlap_index = 0
        for i in range(1, min(len(last_words), len(current_words)) + 1):
            if last_words[-i:] == current_words[:i]:
                overlap_index = i
        result[key] += " " + " ".join(current_words[overlap_index:]).strip()
        last_words = current_words

    transcript = ""
    for key in sorted(result.keys()):
        hh = (key % 3600) // 60
        mm = key % 60
        timecode = f"{hh:02d}:{mm:02d}:00"
        data = f"{timecode} {result[key].strip()}"
        transcript += data + "\n"

    for line in transcript.splitlines():
        timecode = line[:8]
        text = line[9:]
        content = f"Video Name: {video_name}+\nYouTube Channel: {channel_name}\nPublish Date: {publish_date}\nYouTube-ID: {video_id} \nTimecode: {timecode} \nText: {text}"
        yield {
            "Id": re.sub("[^0-9a-zA-Z_-]","_",f"{video_id}-{timecode}"),
            "Text": content,
            "CreatedAt": publish_date,
            "ChannelName": channel_name,
            "VideoName": video_name,
            "Vector": openai.Embedding.create(engine="text-embedding-ada-002", input=text)["data"][0]["embedding"],
         }


def index(embedding_data, acs_key, acs_instance, batch_size=1000):
    search_client = SearchClient(endpoint=f"https://{acs_instance}.search.windows.net/",
                                    index_name="embeddings",
                                    credential=AzureKeyCredential(acs_key))

    current_index = 0
    current_batch = []
    for embedding in embedding_data:
        current_batch.append(embedding)
        current_index += 1
        if current_index % batch_size == 0:
            report_status(search_client, current_batch)
            current_batch = []

    if len(current_batch) > 0:
        report_status(search_client, current_batch)

def report_status(search_client, batch):
    results = search_client.upload_documents(documents=batch)
    successful_uploads = sum(1 for result in results if result.succeeded)
    print(f"\tIndexed {len(results)}, {successful_uploads} succeeded")


if __name__ == "__main__":
    video_id = sys.argv[1]  # get video_id from command-line arguments

    openai.api_type = "azure"
    openai.api_key = os.environ.get('AZURE_OPENAI_API_KEY') 
    openai.api_base = os.environ.get('AZURE_OPENAI_ENDPOINT')
    openai.api_version = "2022-12-01"

    # setup azure cognitive search
    acs_key = os.environ.get('ACS_KEY')
    acs_instance = os.environ.get('ACS_INSTANCE')

    # initialize search index
    initialize_search_index(acs_key, acs_instance)

    # generate embeddings
    embeddings = create_embeddings(video_id)

    # index embeddings
    index(embeddings, acs_key=acs_key, acs_instance=acs_instance)