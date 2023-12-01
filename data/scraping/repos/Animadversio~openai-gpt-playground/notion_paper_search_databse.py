import os
from notion_client import Client
from notion_tools import print_entries
import arxiv

database_id = "d3e3be7fc96a45de8e7d3a78298f9ccd"
notion = Client(auth=os.environ["NOTION_TOKEN"])
arxiv_filter = {"property": "Link", "url": {"contains": "arxiv"}}
all_results = []
# Keep querying pages of results until there are no more
next_page = None
while True:
    # Query the database with the filter criterion and the next page URL
    results = notion.databases.query(
        **{
            "database_id": database_id,
            "filter": arxiv_filter,
            "page_size": 100,
            "start_cursor": next_page
        }
    )
    # Add the current page of results to the list
    all_results.extend(results["results"])
    # Check if there are more pages of results
    if not results["has_more"]:
        break
    # Get the URL of the next page of results
    next_page = results["next_cursor"]
    print(len(all_results))

print_entries(all_results)
#%%
from tqdm import tqdm, trange
abstract_dict = {}
for entry in tqdm(all_results):
    entry_id = entry["id"]
    title = entry["properties"]["Name"]["title"][0]["plain_text"]
    arxiv_id = entry["properties"]["Link"]["url"].split("/")[-1]
    try:
        arxiv_entry = next(arxiv.Search(id_list=[arxiv_id]).results())
        title = arxiv_entry.title
        abstract = arxiv_entry.summary
        author_names = [author.name for author in arxiv_entry.authors]
        abstract_dict[arxiv_id] = (title, author_names, abstract)
    except arxiv.arxiv.HTTPError:
        print(f"Error with {arxiv_id}")
#%%
for arxiv_id in tqdm(["2212.01577",
                      "2209.00588",
                      "1703.01161",
                      "2012.02733v2",
                      "2104.13963",
                      "1901.06523",
                      "1812.10912",
                      "1812.04948v3",
                      "2004.02546v1",]):
    arxiv_entry = next(arxiv.Search(id_list=[arxiv_id]).results())
    title = arxiv_entry.title
    abstract = arxiv_entry.summary
    author_names = [author.name for author in arxiv_entry.authors]
    abstract_dict[arxiv_id] = (title, author_names, abstract)
#%%

# embed them using OpenAI API
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
openai.api_key = os.environ["OPENAI_API_KEY"]
#%%
embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=, embedding_function=embeddings)
#%%
vectordb.add_texts(abstract_dict)
#%%
