import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'))

from app.api.utils import CHUNK_OVERLAP, CHUNK_SIZE, FLASHCARDS_PROMPTS
from langchain import OpenAI, PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import json




class FlashcardsService:
    
    def __init__(self,text, language="en") -> None:
        self._text = text
        self._language = language
        self._llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"],max_tokens=2000)
        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)
        self._flashcards = None
        self._totalTokensUsed = None
        self._totalCost = None
    

    def getFlashcards(self):
        """
            Return the quiz on the text.
            return:
            quiz: String (the quiz on the text)
            totalTokensUsed: int (the total tokens used to generate the quiz)
            totalCost: int (the total cost of the quiz)
        """
        flashcards, tokensUsed = self.__getFlashcards()
        totalTokensUsed = tokensUsed.completion_tokens + tokensUsed.prompt_tokens
        totalCost = tokensUsed.total_cost
        
        self._flashcards= flashcards
        self._totalTokensUsed = totalTokensUsed
        self._totalCost = totalCost
        
        return flashcards,totalTokensUsed, totalCost


    def printFlashcards(self,fladhcards):
        for card in fladhcards:
            print(card,"\n\n")

    def __getFlashcards(self):
        """
        we chunk the text into smaller pieces and then we generate a flashcards list for each chunk.
        combine the final result into one flashcards list.
        """

        PROMPT = PromptTemplate(template=FLASHCARDS_PROMPTS[self._language], input_variables=["text"])
        textList = [ self._text ]
        document = [Document(page_content=t) for t in textList]
        docs = self._text_splitter.split_documents(document)

        with get_openai_callback() as tokensUsed:
            finalFlashcards = []
            for d in docs:
                flashcards = self._llm(PROMPT.format(text=d.page_content))
                flashcardsSeriable =json.loads(flashcards) 
                for q in flashcardsSeriable:
                    finalFlashcards.append(flashcardsSeriable[q])

        return finalFlashcards, tokensUsed