# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: ploomber
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   ploomber:
#     injected_manually: true
# ---

# %% tags=["parameters"]
# add default values for parameters here

# %% tags=["injected-parameters"]
# Parameters
upstream = {
    "txt_to_parquet": {
        "nb": "/home/ubuntu/Agata/product/get/txt_to_parquet.ipynb",
        "data": "/home/ubuntu/Agata/product/get/tweets.parquet",
    }
}
product = {"nb": "/home/ubuntu/Agata/scripts/exploration/rag_tweet.ipynb"}


# %%
import chromadb
import pandas as pd

# %%
sample = True

df = pd.read_parquet(upstream["txt_to_parquet"]["data"])

if sample:
    df = df.sample(20000)

# %%
collection_name = "Tweets"

# %%
chroma_client = chromadb.PersistentClient()

# try:
#     chroma_client.delete_collection(collection_name)
# except:
#     pass
collection = chroma_client.get_or_create_collection(name=collection_name)

# %%
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# %%
embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"

device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

# %%
print(device)

# %%
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={"device": device},
    encode_kwargs={"device": device, "batch_size": 1024},
    multi_process=False,
)

# %%
docs = ["Uncle Pear", "Mom"]

embeddings = embed_model.embed_documents(docs)

print(
    f"We have {len(embeddings)} doc embeddings, each with "
    f"a dimensionality of {len(embeddings[0])}."
)

# %%
from datetime import datetime
from tqdm.auto import trange, tqdm

metadata = [
    {
        "ID_1": int(df["ID_1"].iloc[i]),
        "ID_2": int(df["ID_2"].iloc[i]),
        "handle": df["handle"].iloc[i],
        # "timestamp":datetime.strptime(df["Date"].iloc[i] + " " + df["Time"].iloc[i], "%Y-%m-%d %H:%M:%S")
    }
    for i in trange(len(df))
]

# %%
docsearch = Chroma.from_texts(
    df["content"],
    embed_model,
    metadatas=metadata,
    collection_name=collection_name,
)

# %%
documents = docsearch.search("Musician", search_type="similarity", k=20)

# %%
import guidance
import transformers
from torch import cuda, bfloat16
import os
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM

# %%
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)

hugging_face_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
model_id = "meta-llama/Llama-2-13b-hf"

llm = guidance.llms.Transformers(
    model_id, quantization_config=bnb_config, temperature=0.1
)

# %%
import guidance

guidance.llm = llm
program = guidance(
    """The uncle pear is one of the most{{gen 'adjectives' stop="\\n-" temperature=0.7 max_tokens=10}}.
In fact, he fought in the battle of {{gen 'battle' stop="\\n-" temperature=0.7 max_tokens=1}} """
)
executed = program()


# %%


# %%
# we can pre-define valid option sets
valid_weapons = ["sword", "axe", "mace", "spear", "bow", "crossbow"]

valid_armors = ["leather", "chainmail", "plate"]

# define the prompt
character_maker = guidance(
    """The following is a character profile for an RPG game in JSON format.
```json
{
    "id": "{{id}}",
    "description": "{{description}}",
    "name": "{{gen 'name' temperature=1.2 stop='"'}}",
    "age": {{gen 'age' pattern='[0-9]+' stop=',' temperature=1.2}},
    "weapon": "{{select 'armor' options=valid_armors}}",
    "weapon": "{{select 'weapon' options=valid_weapons}}",
    "class": "{{gen 'class' temperature=0.7 stop='"'}}",
    "subclass": "{{gen 'class' temperature=0.7 stop='"'}}",
    "mantra": "{{gen 'mantra' temperature=0.7 stop='"'}}",
    "strength": {{gen 'strength' pattern='[0-9]+' stop=',' temperature=0.1}},
    "skill_1":"{{gen 'skill_1' temperature=1.5 stop=[',','"']}}",
    "skill_2":"{{gen 'skill_2' temperature=1.5 stop=[',','"']}}",
    "skill_3":"{{gen 'skill_3' temperature=1.5 stop=[',','"']}}",
    "skill_4":"{{gen 'skill_4' temperature=1.5 stop=[',','"']}}",
    "skill_5":"{{gen 'skill_5' temperature=1.5 stop=[',','"']}}",
    "race":"{{gen 'race' temperature=1.5 stop=[',','"']}}",
    "most_hated_races":"{{gen 'race' temperature=0.1 stop=[',','"']}}"
}```"""
)

# generate a character
result = character_maker(
    id="e1f491f7-7ab8-4dac-8c20-c92b5e7d883d",
    description="A quick and nimble archer.",
    valid_weapons=valid_weapons,
    valid_armors=valid_armors,
    llm=llm,
)

# %%
result["race"].strip('"')

# %%
yes_no = ["Yes", "No"]
program = guidance(
    """Problem: You have a database of tweets. The objective is to find the 5 most liked genre of music.
You can iteratively suggest keywords used to search the tweets database and then perform considerations \
about continoung or stopping.
Keyword 1 : <music>.
SEARCH RESULTS:
1. 'what kind of music  http://myloc.me/4LTTT'
2. 'dancing!'
3. 'WHAT GENRE DO YOU PREFER?'
4. 'the only i thing i can do with music on is dishes. swear to god. thus i have only heard 2 or 3 records in the past 3 years.'
5. 'this music...man this music....what you got goin with the boys up that way?'
Is this enough information to create a list? Answer yer or no: "No"
Then, suggest another keyword.
Keyword 2: <genre>
SEARCH RESULTS:
1. WHAT GENRE DO YOU PREFER?
2. what kind of music  http://myloc.me/4LTTT
3. well, music for simple folk, but that's my opinion.
4. what about artists
5. cool. English lit & lang, Music Tech and Film.
6. kool I'm listening to music and writing lyrics
7. dancing!
Is this enough information to create a list? Answer yer or no: "No"
Keyword 3: <artist>
SEARCH RESULTS:
1. I love EDM
2. I love Rock
3. I love Jazz
4. I love Pop
5. I love EDM
Is this enough information to create a list? Answer yer or no: "Yes"
Suggest now the 5 most liked type of music.
First Suggestion: <{{gen 'keyword' temperature=0.1 stop='>'}}>
Second Suggestion: <{{gen 'keyword' temperature=0.1 stop='>'}}>
Third Suggestion: <{{gen 'keyword' temperature=0.1 stop='>'}}>
Fourth Suggestion: <{{gen 'keyword' temperature=0.1 stop='>'}}>
Fifth Suggestion: <{{gen 'keyword' temperature=0.1 stop='>'}}>
Sixth Suggestion: <{{gen 'keyword' temperature=0.1 stop='>'}}>
Seventh Suggestion: <{{gen 'keyword' temperature=0.1 stop='>'}}>
Eighth Suggestion: <{{gen 'keyword' temperature=0.1 stop='>'}}>
Ninth Suggestion: <{{gen 'keyword' temperature=0.1 stop='>'}}>
Tenth Suggestion: <{{gen 'keyword' temperature=0.1 stop='>'}}>
Which is the best in your opinion? <{{gen 'keyword' temperature=0.1 stop='>'}}>. Why? <{{gen 'keyword' temperature=0.1 stop='>'}}>
Give a detailed explanation. <{{gen 'keyword' temperature=0.1 stop='>'}}>
"""
)
program(yes_no=yes_no)

# %%
documents = docsearch.search("music, genre, artist", search_type="similarity", k=20)
texts = [document.page_content for document in documents]
for i in range(len(texts)):
    print(f"{i+1}. {texts[i]}")
    if i > 5:
        break
