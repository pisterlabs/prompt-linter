from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'))
import os
from langchain import OpenAI, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.api.utils import CHUNK_SIZE , CHUNK_OVERLAP , SUMMARY_PROMPTS
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback
from langchain.chains.summarize import load_summarize_chain


class SummaryService:

    def __init__(self, longText, language="en"):
        self._longText = longText
        self._language = language
        self._llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"],max_tokens=2000)
        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)
        self._summary = None
        self._totalTokensUsed = None
        self._totalCost = None

    def getLangChainSummary(self):
        
        """
            Return the summary of the text.
            return:
            summary: String (the summary of the text)
            totalTokensUsed: int (the total number of tokens used to generate the summary)
            totalCost: int (the total cost of the summary)
        """
       
        summary, tokensUsed = self.__langChainSummary()
        totalTokensUsed = tokensUsed.completion_tokens + tokensUsed.prompt_tokens
        totalCost = tokensUsed.total_cost

        self._summary = summary
        self._totalTokensUsed = totalTokensUsed
        self._totalCost = totalCost
        
        return summary, totalTokensUsed, totalCost
    

    def __langChainSummary(self):
        """
        Provides a summary using langchain map_reduce chain.
        """
        longTextList = [ self._longText ]
        document = [Document(page_content=t) for t in longTextList]
        docs = self._text_splitter.split_documents(document)
        PROMPT = PromptTemplate(template=SUMMARY_PROMPTS[self._language], input_variables=["text"])
        with get_openai_callback() as tokensUsed:
            chain = load_summarize_chain(llm=self._llm, combine_prompt=PROMPT,map_prompt=PROMPT, chain_type="map_reduce")
            # summary =  chain.run(docs)
            summary =  chain({"input_documents": docs},return_only_outputs=True)["output_text"]

        return summary, tokensUsed
    