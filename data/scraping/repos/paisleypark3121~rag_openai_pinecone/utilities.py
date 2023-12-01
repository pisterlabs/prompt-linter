from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI

import pinecone
import time
from tqdm.auto import tqdm 

from datasets import load_dataset

def simple_chat(llm):

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hi AI, how are you today?"),
        AIMessage(content="I'm great thank you. How can I help you?"),
        HumanMessage(content="I'd like to understand string theory.")
    ]

    #RESPONSE
    res = llm(messages)
    print(res)
    messages.append(res)

    #REQUEST
    prompt = HumanMessage(
        content="Why do physicists believe it can produce a 'unified theory'?"
    )
    messages.append(prompt)

    #RESPONSE
    res = llm(messages)
    print(res.content)
    messages.append(res)

    #REQUEST
    prompt = HumanMessage(
        content="What is so special about Llama 2?"
    )
    messages.append(prompt)

    #RESPONSE
    res = llm(messages)
    print(res.content)
    messages.append(res)

    #REQUEST
    prompt = HumanMessage(
        content="Can you tell me about the LLMChain in LangChain?"
    )
    messages.append(prompt)

    #RESPONSE
    res = llm(messages)
    print(res.content)
    messages.append(res)

def create_or_load_pinecone_index(dataset,index_name,embed_model):

    pinecone.init()

    if index_name not in pinecone.list_indexes():
        
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine'
        )
        while not pinecone.describe_index(index_name).status['ready']:
            time.sleep(1)
        
            data = dataset.to_pandas()  

        batch_size = 100

        for i in tqdm(range(0, len(data), batch_size)):
            i_end = min(len(data), i+batch_size)
            batch = data.iloc[i:i_end]
            ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
            texts = [x['chunk'] for _, x in batch.iterrows()]
            embeds = embed_model.embed_documents(texts)
            metadata = [
                {'text': x['chunk'],
                'source': x['source'],
                'title': x['title']} for i, x in batch.iterrows()
            ]
            index.upsert(vectors=zip(ids, embeds, metadata))

    return pinecone.Index(index_name)

def augment_prompt(query: str,results):
    source_knowledge = "\n".join([x.page_content for x in results])
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

def get_result(query,messages,vectorstore,llm):
    results=vectorstore.similarity_search(query, k=3)
    # print(results)

    prompt = HumanMessage(
        content=augment_prompt(query=query,results=results)
    )
    messages.append(prompt)
    res = llm(messages)
    # print(res.content)
    return res.content

def arxiv_papers(llm):

    embed_model = OpenAIEmbeddings()
    # texts = [
    #     'this is the first chunk of text',
    #     'then another second chunk of text is here'
    # ]

    # res = embed_model.embed_documents(texts)
    # print(len(res))
    # print(len(res[0]))

    dataset = load_dataset(
        "jamescalam/llama-2-arxiv-papers-chunked",
        split="train"
    )
    #print(dataset[0])

    pinecone.init()
    index_name = 'llama-2-rag'
    index=create_or_load_pinecone_index(
        dataset=dataset,
        index_name=index_name,
        embed_model=embed_model)
    # print(index.describe_index_stats())

    text_field = "text"
    vectorstore = Pinecone(
        index, embed_model.embed_query, text_field
    )

    messages = [
        SystemMessage(content="You are a helpful assistant. That can answer questions based on a specific give context"),
    ]

    query = "What is so special about Llama 2?"
    res=get_result(
        query=query,
        messages=messages,
        vectorstore=vectorstore,
        llm=llm)
    print(res)

    query = "what safety measures were used in the development of llama 2?"
    res=get_result(
        query=query,
        messages=messages,
        vectorstore=vectorstore,
        llm=llm)
    print(res)