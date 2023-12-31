from langchain import OpenAI 
from langchain.llms import AzureOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import textwrap

def summarize_webpage(text):
    # llm = OpenAI(temperature=0)

    deployment = "text-davinci-003-deployment"
    # llm = AzureOpenAI(deployment_name="text-davinci-003-deployment", model_name="text-davinci-003")
    llm = AzureOpenAI(deployment_name=deployment, model_name="text-davinci-003")

    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(text)
    print(len(texts))
    docs = [Document(page_content=t) for t in texts[:4]]
    chain = load_summarize_chain(llm, 
                             chain_type="map_reduce", verbose=True)


    output_summary = chain.run(docs)
    wrapped_text = textwrap.fill(output_summary, width=100)
    print(wrapped_text)

    return wrapped_text