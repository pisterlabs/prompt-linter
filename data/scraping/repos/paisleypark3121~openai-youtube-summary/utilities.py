from urllib.parse import urlparse, parse_qs

from typing import Optional, Tuple
from langchain.document_loaders import YoutubeLoader

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


def get_youtube_transcript(url,language_code="en"):
    loader = YoutubeLoader.from_youtube_url(
        url,        
        add_video_info=True,
        language=[language_code, "id"],
        translation=language_code,)
    docs=loader.load()
    return docs[0].page_content


def get_summary(llm,info_about_me,transcript):
    
    # summary_template = """Please summarize the following transcript in a form of a list with key takeaways.\
    # Tailor the summary for the person who is {info_about_me}.\

    # Transcript: {transcript}
    # """

    summary_template = """Per favore fai un riassunto del seguente Transcript.\
    Cerca di adattare il riassunto per una persona con le seguenti caratteristiche: {info_about_me}.\

    Transcript: {transcript}
    """

    summary_prompt = PromptTemplate(
        input_variables=["transcript", "info_about_me"], 
        template=summary_template)
    summary_chain = LLMChain(
        llm=llm, 
        prompt=summary_prompt, 
        output_key="summary")

    return summary_chain.predict(transcript=transcript,info_about_me=info_about_me)

    