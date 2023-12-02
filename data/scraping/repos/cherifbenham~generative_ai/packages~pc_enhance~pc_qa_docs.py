import openai
import os
from datasets import load_dataset
from tqdm.auto import tqdm
import pinecone
from time import sleep
from datasets import load_dataset

#dataset name
dataset='jamescalam/youtube-transcriptions'

#new index to create
index_name = 'openai-youtube-transcriptions'

#openai credentials
openai.api_key = os.getenv("CHATGPT_API_KEY")
embed_model = "text-embedding-ada-002"
dimension_embedding=1536

#pinecone variables
environment="us-east1-gcp"
api_key="8ff9b8af-efae-48f0-985b-3298de8e36c9"
limit = 3750
query="Which training method should I use for sentence transformers when I only have pairs of related sentences?"

def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()

def load_data(dataset):
    data = load_dataset(dataset, split='train')
    return data

def merge_snippets(data):
    new_data = []

    window = 20  # number of sentences to combine
    stride = 4  # number of sentences to 'stride' over, used to create overlap

    for i in tqdm(range(0, len(data), stride)):
        i_end = min(len(data)-1, i+window)
        if data[i]['title'] != data[i_end]['title']:
            # in this case we skip this entry as we have start/end of two videos
            continue
        text = ' '.join(data[i:i_end]['text'])
        # create the new merged dataset
        new_data.append({
            'start': data[i]['start'],
            'end': data[i_end]['end'],
            'title': data[i]['title'],
            'text': text,
            'id': data[i]['id'],
            'url': data[i]['url'],
            'published': data[i]['published'],
            'channel_id': data[i]['channel_id']
        })
    return new_data

def create_embedding(embed_model, query):
    res = openai.Embedding.create(
        input=[query], #for example: this list (, comma separated) "Sample document text goes here", "there will be several phrases in each batch", 
        engine=embed_model # for example: embed_model = "text-embedding-ada-002"
    )
    return res
        
def initialize_index(index_name, api_key, environment):
    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key=api_key,
        environment=environment
    )
    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
        # if does not exist, create index
        pinecone.create(
            index_name,
            dimension=dimension_embedding,
            metric='cosine',
            metadata_config={'indexed': ['channel_id', 'published']}
        )
    
    # connect to index
    index = pinecone.Index(index_name)
    return index

def populate_index(index, new_data):
    batch_size = 100  # how many embeddings we create and insert at once
    for i in tqdm(range(0, len(new_data), batch_size)):
        # find end of batch
        i_end = min(len(new_data), i+batch_size)
        meta_batch = new_data[i:i_end]
        # get ids
        ids_batch = [x['id'] for x in meta_batch]
        # get texts to encode
        texts = [x['text'] for x in meta_batch]
        # create embeddings (try-except added to avoid RateLimitError)
        try:
            res = openai.Embedding.create(input=texts, engine=embed_model)
        except:
            done = False
            while not done:
                sleep(5)
                try:
                    res = openai.Embedding.create(input=texts, engine=embed_model)
                    done = True
                except:
                    pass
        embeds = [record['embedding'] for record in res['data']]
        # cleanup metadata
        meta_batch = [{
            'start': x['start'],
            'end': x['end'],
            'title': x['title'],
            'text': x['text'],
            'url': x['url'],
            'published': x['published'],
            'channel_id': x['channel_id']
        } for x in meta_batch]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        # upsert to Pinecone
        index.upsert(vectors=to_upsert)     
    return index

def retrieve(query, index, embed_model):
    res = create_embedding(embed_model, query)

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # initialize the index - needs to be done once outside this function
    # index=initialize_index(index_name, res, environment)

    #populate the index - needs to be done once outside this function
    # index=populate_index(index,new_data)

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    contexts = [x['metadata']['text'] for x in res['matches']]

    # build our prompt with the retrieved contexts included
    prompt_start = ("Answer the question based on the context below.\n\n"+"Context:\n")
    prompt_end = (f"\n\nQuestion: {query}\nAnswer:")

    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (prompt_start +
                        "\n\n---\n\n".join(contexts[:i-1]) +
                        prompt_end)
            break
        elif i == len(contexts)-1:
            prompt = (prompt_start +
                        "\n\n---\n\n".join(contexts) +
                        prompt_end)
    return prompt