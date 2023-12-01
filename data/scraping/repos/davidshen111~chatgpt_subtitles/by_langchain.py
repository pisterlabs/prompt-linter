import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

from dotenv import load_dotenv, find_dotenv

def fetch_by_langchain_mapreduce(converted_subtitles):

    openai_api_key  = os.environ['OPENAI_API_KEY']
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
    docs = [Document(page_content=t) for t in converted_subtitles]

    template_str = """Your task is to generate an overall summary for the following contents:
    {text}
    The output should be a text in UTF-8 format, written in Chinese."""
    COMMON_PROMPT = PromptTemplate(input_variables=["text"], template=template_str)

    # We can define two prompt templates, one for map_prompt and another one for combine_prompt. We take the simple way for this case. 
    chain = load_summarize_chain(llm, 
                                 chain_type="map_reduce", 
                                 return_intermediate_steps=True, 
                                 map_prompt=COMMON_PROMPT, 
                                 combine_prompt=COMMON_PROMPT,
                                 verbose=True)
    output_summary = chain({"input_documents": docs}, return_only_outputs=True)
    return output_summary['output_text']


def fetch_by_langchain_refine(converted_subtitles):

    openai_api_key  = os.environ['OPENAI_API_KEY']
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
    docs = [Document(page_content=t) for t in converted_subtitles]

    refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary\n"
    "If the context isn't useful, return the original summary."
    "The output should be a text in UTF-8 format, written in Chinese."
    )
    
    REFINE_PROMPT = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template,
    )
    
    prompt_template = """Your task is to generate a summary for the following contents:       
    "{text}"
    "The summary should be a text in UTF-8 format, written in Chinese."
    SUMMARY:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    
    chain = load_summarize_chain(llm, 
                                 chain_type="refine", 
                                 return_intermediate_steps=True, 
                                 question_prompt=PROMPT, 
                                 refine_prompt=REFINE_PROMPT,
                                 verbose=True)
    output_summary = chain({"input_documents": docs}, return_only_outputs=True)
    return output_summary['output_text']

