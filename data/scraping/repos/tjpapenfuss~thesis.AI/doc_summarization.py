from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
import config

import uuid

def encode_to_guid(input_string):
    # Generate a UUID version 3 using the input string as the namespace
    namespace_uuid = uuid.uuid3(uuid.NAMESPACE_DNS, input_string)

    # Convert the UUID to a string representation
    guid_string = str(namespace_uuid)

    return guid_string

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def summarize_doc(document):
    openai_api_key = config.api_key
    llm=ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=0, separators=[" ", ",", "\n"])

    #text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.create_documents([document])

    map_prompt = """
    You will be given a website. 
    Your goal is to give a summary of this website so that a reader will have a full understanding.
    Your response should be at least three paragraphs and fully encompass what was said in the passage.

    ```{text}```
    FULL SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    map_chain = load_summarize_chain(llm=llm,
                                chain_type="stuff",
                                prompt=map_prompt_template)
    return map_chain.run(texts)

