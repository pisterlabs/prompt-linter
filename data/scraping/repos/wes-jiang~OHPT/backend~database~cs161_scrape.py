from bs4 import BeautifulSoup
import featureform as ff
from featureform import local
import requests
import re 
import time
import pandas as pd
import openai

from sentence_transformers import SentenceTransformer

# steps: 
# request https://textbook.cs161.org/
# obtain links to all the subsections (security principles, memory safety, etc)
# for each subsection
#   iterate through the topics in TOCs and get links to them
#   for each topic
#       turn into chunks

##############################
# PHASE 0: SET UP DATABASE/API KEYS
##############################


# pinecone
pinecone = ff.register_pinecone(
    name="pinecone",
    project_id="____",
    environment="____",
    api_key="_____",
)

# openai
openai.organization = "org-V70xAGNCjfzw012seLYRWNTJ"
openai.api_key = "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"

##############################
# PHASE 1: SCRAPE DATA
##############################

home_url = "https://textbook.cs161.org/"

#  request HTML from home url
homepage = requests.get(home_url).text

# parse the html to get a BeautifulSoup obj
doc = BeautifulSoup(homepage, "html.parser")

# zoom in on the navbar
navbar_div = doc.find(class_="nav-list")

# create dictionary for storing chunks
textbook_dict = {"PK": [], "Text": []}
def addToDict(dn, pk, text):
    dn["PK"].append(pk)
    dn["Text"].append(text)

# iterate through all subpage <a> tags and scrape from them too
subpage_links = navbar_div.find_all(class_="nav-list-link")
for link in subpage_links:
    sub_url = link["href"]

    content_url = f"https://textbook.cs161.org{sub_url}"
    print(content_url)
    
    # scrape the content page
    content_page = requests.get(content_url).text
    section_doc = BeautifulSoup(content_page, "html.parser")

    content = section_doc.find(class_="main-content")

    # find all paragraphs within the content text
    paragraphs = content.find_all("p")
    # iterate through paragraphs, create chunks and add to dictionary
    for i in range(len(paragraphs) - 1):
        pk = sub_url + str(i)
        text = paragraphs[i].text + paragraphs[i+1].text
        addToDict(textbook_dict, pk, text)

    print("database size: ", len(textbook_dict["PK"]))

    print()
    print("-------------------")
    print()
    

########################
# PHASE 2: data processing
########################

# turn into a dataframe
textbook_df = pd.DataFrame(textbook_dict)

client = ff.Client(local=True)

client.dataframe(textbook_df)

@local.df_transformation(inputs=[textbook_df])
def vectorize_excerpts(df):
    # vectorize chunks using a sentence transformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["Text"].tolist())
    textbook_df["Vector"] = embeddings.tolist()



# upload database to pinecone
# TODO


textbook_df.to_csv('./test.csv')