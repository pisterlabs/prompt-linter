from langchain.llms import OpenAI
from langchain.llms import AzureOpenAI
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

class PDFSummarizer:
    def setOpenAIConfig(self, openai_api_key, model_name: str="gpt4", temperature=0.0):
        self.llm = OpenAI(api_key=openai_api_key,
                          api_type="openai", 
                          api_version = '2020-11-07',
                          api_base = "https://api.openai.com/v1",
                          temperature=temperature)

    def setAzureOpenAIConfig(self, azure_openai_api_key: str, azure_openai_api_base: str, azure_openai_deployment_name: str, 
                             model_name: str="gpt4", temperature=0.0):
        openai_api_version="2023-05-15"
        openai_api_type="azure"
        self.llm = AzureOpenAI(
            api_type=openai_api_type,
            api_key=azure_openai_api_key,
            api_base=azure_openai_api_base,
            deployment_name=azure_openai_deployment_name,
            model=model_name,
            temperature=temperature,
            api_version=openai_api_version)
    
    def read_contents(self, url):
        loader = OnlinePDFLoader(url)
        pages = loader.load()
        text = ""
        for page in pages:
            text += page.page_content
    
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts[:3]]
        return docs

        
    def summarize_contents(self, url):
        docs = self.read_contents(url)
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        return chain.run(docs)