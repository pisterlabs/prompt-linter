import os
from pprint import pprint

from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain import HuggingFaceHub, OpenAI
from langchain.llms import BaseLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback

"""
Recommended LLMs for each task

Summarization task - facebook/bart-large-cnn
Question Answering task - deepset/roberta-base-squad2
"""


# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")


def summarise_transcript_with_open_ai(youtube_url: str):
    """
    Very simple example that downloads a transcript from YouTube and summarises it with the OpenAI API
    """
    print("###############")
    print("# Running summarise_transcript_with_open_ai function")
    print("###############")
    # IMPROVEMENT: Cache the transcript
    print("loading transcript...")
    loader = YoutubeLoader.from_youtube_url(
        youtube_url,
        add_video_info=True,
    )
    docs = loader.load()
    print("# Docs: ", len(docs))

    llm = OpenAI(temperature=0)

    total_tokens = 0
    for doc in docs:
        total_tokens += llm.get_num_tokens(doc.page_content)
    print("Total tokens for docs: ", total_tokens)

    max_prompt = llm.modelname_to_contextsize(llm.model_name) - llm.max_tokens
    chain_type = "stuff"
    print("The max. context size is: ", max_prompt)

    print(f"{total_tokens} > {max_prompt} ?")
    if total_tokens > max_prompt:
        chain_type = "map_reduce"
        print("We need to split the transcript")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        print("Splitting docs...")
        transcript = text_splitter.split_documents(docs)
        # IMPROVEMENT: Create logic to determine if we need to use a chain type = map_reduce or stuff
        print("# Split docs: ", len(transcript))
    else:
        transcript = docs

    chain = load_summarize_chain(llm, chain_type=chain_type, verbose=False)
    with get_openai_callback() as token_info:
        summary = chain.run(transcript)
        print(token_info)

    pprint(f"Summary: {summary}")

def load