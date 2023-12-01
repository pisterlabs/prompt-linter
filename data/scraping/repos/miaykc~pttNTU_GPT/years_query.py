import dataclasses
from dataclasses import dataclass
from pprint import pprint
import os
import weaviate
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
#import json
import csv

@dataclass
class ContentItem:
    media: str  # media source of the post or comment
    content_type: str  # post or comment
    author: str  # author of the post or comment
    post_id: str  # id of the post
    year: str  # year of the post
    board: str  # board of the post
    title: str  # title of the post
    text: str  # text of the post or comment
    rating: str  # rating of the comment
    order: int  # 0 for post, 1, 2, 3, ... for comments
    chunk: int  # if text too long, split into chunks
    total_chunks: int  # total number of chunks


os.environ['WEAVIATE_ADMIN_PASS'] = "weaviate-ultimate-forever-pass"
os.environ['OPENAI_API_KEY'] = "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"

client = weaviate.Client(
    url="http://140.112.147.128:8000",
    auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_ADMIN_PASS"]),
    timeout_config=(5, 30), # (connect timeout, read timeout) # type: ignore
    additional_headers={'X-OpenAI-Api-Key': os.environ["OPENAI_API_KEY"]}
)

# https://weaviate.io/blog/hybrid-search-explained
attributes = [field.name for field in dataclasses.fields(ContentItem)]
print(attributes)
retriever = WeaviateHybridSearchRetriever(
    client=client,
    k=50,
    alpha=0.5,  # weighting for each search algorithm (alpha = 0 (sparse, BM25), alpha = 1 (dense), alpha = 0.5 (equal weight for sparse and dense))
    index_name="ContentItem",
    text_key="text",
    attributes=attributes,  # include these attributes in the 'metadata' field of the search results
)

def y2020_filter(keyword):
    where_filter = {
        "operator": "And",  # And or Or
        "operands": [  # use operands for multiple filters
            {"path": ["content_type"], "operator": "Equal", "valueString": "comment"},
            {"path": ["rating"], "operator": "Equal", "valueString": "pos"},
            {"path": ["author"], "operator": "NotEqual", "valueString": "peterW"},
            {"path": ["year"], "operator":"Equal", "valueString":"2020"}
        ],
    }
    r = retriever.get_relevant_documents(keyword, where_filter=where_filter)
    pprint(r)
    return r

def y2021_filter(keyword):
    where_filter = {
        "operator": "And",  # And or Or
        "operands": [  # use operands for multiple filters
            {"path": ["content_type"], "operator": "Equal", "valueString": "comment"},
            {"path": ["rating"], "operator": "Equal", "valueString": "pos"},
            {"path": ["author"], "operator": "NotEqual", "valueString": "peterW"},
            {"path": ["year"], "operator":"Equal", "valueString":"2021"}
        ],
    }
    r = retriever.get_relevant_documents(keyword, where_filter=where_filter)
    pprint(r)
    return r

def y2022_filter(keyword):
    where_filter = {
        "operator": "And",  # And or Or
        "operands": [  # use operands for multiple filters
            {"path": ["content_type"], "operator": "Equal", "valueString": "comment"},
            {"path": ["rating"], "operator": "Equal", "valueString": "pos"},
            {"path": ["author"], "operator": "NotEqual", "valueString": "peterW"},
            {"path": ["year"], "operator":"Equal", "valueString":"2022"}
        ],
    }
    r = retriever.get_relevant_documents(keyword, where_filter=where_filter)
    pprint(r)
    return r

def y2023_filter(keyword):
    where_filter = {
        "operator": "And",  # And or Or
        "operands": [  # use operands for multiple filters
            {"path": ["content_type"], "operator": "Equal", "valueString": "comment"},
            {"path": ["rating"], "operator": "Equal", "valueString": "pos"},
            {"path": ["author"], "operator": "NotEqual", "valueString": "peterW"},
            {"path": ["year"], "operator":"Equal", "valueString":"2023"}
        ],
    }
    r = retriever.get_relevant_documents(keyword, where_filter=where_filter)
    pprint(r)
    return r

def y2020to2023_filter(keyword):
    where_filter = {
        "operator": "And",  # And or Or
        "operands": [  # use operands for multiple filters
            {"path": ["content_type"], "operator": "Equal", "valueString": "comment"},
            {"path": ["rating"], "operator": "Equal", "valueString": "pos"},
            {"path": ["author"], "operator": "NotEqual", "valueString": "peterW"}
        ],
    }
    r = retriever.get_relevant_documents(keyword, where_filter=where_filter)
    pprint(r)
    return r

def data_cleaner(r):
    post_ids = []

    for doc in r:
        post_ids.append(doc.metadata['post_id'])
    filtered_post_ids = list(set(post_ids))

    result={}
    for doc in r:
        if doc.metadata['post_id'] in filtered_post_ids:
            if doc.metadata['post_id'] in result:
                result[doc.metadata['post_id']] += '' + doc.page_content
            else:
                result[doc.metadata['post_id']] = doc.page_content

#    with open("NTU_library.json","w") as json_file:
#        json.dump(result, json_file)

    with open("NTU_library.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['post_id', 'page_content'])
        for post_id, page_content in result.items():
            writer.writerow([post_id, page_content])

    for post_id, page_content in result.items():
        print(f"post_id: {post_id}, page_content: {page_content}")
    

def merge_page_content(input_file, output_file):
    with open(input_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # 跳過標題列

        merged_content = []

        for row in reader:
            page_content = row[1]
            merged_content.append(page_content)

    with open(output_file, 'w') as txt_file:
        for content in merged_content:
            txt_file.write(f"{content}\n")