import re

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def youtube_video_url_is_valid(url: str) -> bool:
    pattern = r'^https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)(\&ab_channel=[\w\d]+)?$'
    match = re.match(pattern, url)
    return match is not None


def find_insights(api_key: str, url: str) -> str:
    try:
        loader = YoutubeLoader.from_youtube_url(url)
        transcript = loader.load()
    except Exception as e:
        return f"Error while loading YouTube video and transcript: {e}"

    try:
        llm = OpenAI(temperature=0.6, openai_api_key=api_key)
        prompt = PromptTemplate(
            template="""Summarize the youtube video whose transcript is provided within backticks \
            ```{text}```
            """, input_variables=["text"]
        )
        combine_prompt = PromptTemplate(
            template="""Combine all the youtube video transcripts  provided within backticks \
            ```{text}```
            Provide a concise summary between 8 to 10 sentences.
            """, input_variables=["text"]
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        text = text_splitter.split_documents(transcript)
        chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False,
                                     map_prompt=prompt, combine_prompt=combine_prompt)
        answer = chain.run(text)
    except Exception as e:
        return f"Error while processing and summarizing text: {e}"

    return answer.strip()
