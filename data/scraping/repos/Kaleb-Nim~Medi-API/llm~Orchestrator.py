from dotenv import load_dotenv
import pinecone
import openai
import os
import json
from time import time 
import tqdm
# Typing
from typing import List, Dict, Any, Optional, Union, Tuple

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone.index import Index
from pinecone.index import UpsertResponse
from langchain.chains import LLMChain
from llm.chains import output_chain
# Data processing stuff
import pandas as pd

from PineconeUtils.Indexer import Indexer,DataEmbedding
from PineconeUtils.Queryer import PineconeQuery

class Orchestrator:
    """Main wrapper class to handle both Pinecone Query doc and Langchain QA docs"""

    def __init__(self,pineconeQuery:PineconeQuery,chain:LLMChain):
        self.pineconeQuery = pineconeQuery
        self.chain = chain

    @staticmethod
    def extractQuestionsFromDocs(matched_docs:list[dict]) -> list[dict]:
        """Filter the metadata from the matched docs
        Args:
            matched_docs (list[dict]): List of matched documents
        Returns:
            list[list[str]]: List of questions 
        """
        questions_list = []
        for doc in matched_docs:
            metadata = doc["metadata"]
            text = metadata["questions"]
            questions_list.append(text)
        return questions_list

    @staticmethod
    def formatQuestionPrompt(questions_list:list[list[str]]) -> str:
        """Format the questions to be used as prompt for GPT-3
        Args:
            questions (list[str]): List of questions
        Returns:
            str: Formatted questions
        """

        question_prompt = ""
        for i,questions in enumerate(questions_list):
            question_prompt += f"---\n"
            for e,question in enumerate(questions):
                question_prompt += f"{i+1}.{e+1}) {question}\n "
        
        return question_prompt

    
    def findRelevantQuestion(self,question:str)->dict:
        """
        Find the most relevant question from the Langchain QA docs, if theres no relevant question, return false

        Args:
            question (str): Question to search
        
        Returns:
            {
                "isValidQuestion": true,
                "matched_question": "tube feeding impacts",
                "question_list_index": 2
            }
        """
        docs = self.pineconeQuery.query(question,top_k=5)

        questions = Orchestrator.extractQuestionsFromDocs(docs)
        question_prompt = Orchestrator.formatQuestionPrompt(questions)

        # Get the answer from the chain
        answer = self.chain.run(questions= question_prompt,user_question=question)
        
        return answer