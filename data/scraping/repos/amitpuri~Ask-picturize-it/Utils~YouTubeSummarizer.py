from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.llms import AzureOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

class YouTubeSummarizer:
    def setOpenAIConfig(self, openai_api_key, model_name: str="gpt4", temperature=0.0):
        self.llm = OpenAI(api_key=openai_api_key,
                          api_type="openai", 
                          api_version = '2020-11-07',
                          temperature=temperature,
                          api_base = "https://api.openai.com/v1")

    def setAzureOpenAIConfig(self, azure_openai_api_key: str, 
                             azure_openai_api_base: str, azure_openai_deployment_name: str, 
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

    def transcribe(self, url):
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        result = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        texts = text_splitter.split_documents(result)
        return texts

    def summarize(self, url):
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        result = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        texts = text_splitter.split_documents(result)
        chain = load_summarize_chain(self.llm, chain_type="map_reduce", verbose=True)
        return chain.run(texts[:4])
