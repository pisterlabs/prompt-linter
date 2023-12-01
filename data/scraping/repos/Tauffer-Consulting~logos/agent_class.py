from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import models
import openai
from config import config
import cohere
import os
from dotenv import load_dotenv
from io import StringIO
import threading
import queue

load_dotenv()


class Agent():
    def __init__(self) -> None:
        self.tools = [
            Tool(
                name="search_internal_knowledge_base",
                func=lambda question: self.qdrant_search(question),
                description="""Useful for searcing the internal knowledge base about general.
Only use this tool if no other specific search tool is suitable for the task."""
            ),
            Tool(
                name="search_internal_knowledge_base_for_specific_author",
                func=lambda question: self.search_by_author(question),
                description="""Only use this tool when the name of the specific author is known and mentioned in the question.
Use this tool for searching information about this specific author.
If the name of the author is not explicitly mentioned in the original question DO NOT USE THIS TOOL.
The input to this tool should contain the name of the author and the information you are trying to find. 
Input template: 'AUTHOR: name of the author INFORMATION: the information you are searching for in the form of a long and well composed question'"""
            ),
            Tool(
                name="search_internal_knowledge_base_for_specific_document_title",
                func=lambda question: self.search_by_title(question),
                description="""Use this only when you are searching for information about one specific document title 
and you know this document's title. Do not use this if you do not know the document's title. 
Create an input with the title of the document and the information you are searching for them.
Input template: 'TITLE: title of the document INFORMATION: the information you are searching for in the form of a long and well composed question'"""
            ),
            Tool(
                name="comparison_and_summarize_tool",
                func=lambda question: self.compare_and_summarize(question),
                description="""Useful for gathering already collected information and making a comparison.
Only use this tool when you have already collect specific information and now just needs to compare and summarize it as a final answer.
For example, if you have already collected information about two different authors and now you want to compare them and summarize the comparison.
For example: 'COLLECTED INFORMATION: AUTHOR1 said things wer like this and that, AUTHOR2 said the things were like that and this. QUESTION: what are the differences or similarities between the two authors?'
Input template: 'COLLECTED INFORMATION: all the information you've collected so far QUESTION: the original question you are trying to answer'"""
            ),
        ]
        self.agent = initialize_agent(
            tools=self.tools, 
            llm=OpenAI(
                temperature=0.1,
                model_name="gpt-3.5-turbo",
                # model_name="gpt-4",
            ), 
            agent="zero-shot-react-description", 
            max_iterations=6,
            verbose=True,
        )
        self.qdrant_answers = []
        self.language = ''


    def get_openai_response(self, qdrant_answer, question):
        prompt = ""
        for r in qdrant_answer:
            prompt += f"""excerpt: author: {r.payload.get('author')}, title: {r.payload.get('title')}, text: {r.payload.get('text')}\n"""
        if len(prompt) > 10000:
            prompt = prompt[0:10000]
        prompt += f"""
Given the excerpts above, answer the following question in {self.language}:
Question: {question}"""
        messages = [{"role": "user", "content": prompt}]
        openai_model = 'gpt-3.5-turbo'
        openai_answer = openai.ChatCompletion.create(
            model=openai_model,
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
        )
        if not openai_answer or not openai_answer.choices:
            return "No answer found"
        return str(openai_answer.choices[0].message.content)
    

    def compare_and_summarize(self, text):
        prompt = f"""Based on the collected information, answer the question in {self.language}:
{text}"""
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
            # frequency_penalty=0.0,
            # presence_penalty=0.0,
            # stop=["\n"]
        )
        return str(response.choices[0].message.content)


    def get_cohere_embeddings(self, texts: list, model: str = None) -> list:
        cohere_client = cohere.Client(config.COHERE_API_KEY)
        if model is None:
            model = 'multilingual-22-12'
        embeddings = cohere_client.embed(
            texts=texts,
            model=model,
        )
        embeddings = [float(e) for e in embeddings.embeddings[0]] 
        return embeddings


    def get_qdrant_response(self, question, limit: int = 8):
        embeddings = self.get_cohere_embeddings(texts=[question])
        db_client = QdrantClient(
            host=config.QDRANT_HOST,
            api_key=config.QDRANT_API_KEY,
        )
        response = db_client.search(
            collection_name="hackathon_collection",
            query_vector=embeddings,
            limit=limit,
        )
        self.qdrant_answers.extend(response)
        return response


    def qdrant_search(self, question):
        qdrant_answer = self.get_qdrant_response(question)
        return self.get_openai_response(qdrant_answer, question)


    def get_qdrant_response_by_filter(self, question, key, value, limit: int = 8):
        embeddings = self.get_cohere_embeddings(texts=[question])
        db_client = QdrantClient(
            api_key=os.environ.get('QDRANT_API_KEY'),
            host=os.environ.get('QDRANT_HOST')
        )
        response = db_client.search(
            collection_name="hackathon_collection",
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(
                                value=value
                            ) 
                        )
                    ]
                ),
            query_vector=embeddings,
            limit=limit
        )
        self.qdrant_answers.extend(response)
        return response


    def search_by_author(self, question):
        author_info, question_info = question.split('AUTHOR:', 1)[1].split('INFORMATION:', 1)
        author = author_info.strip().lower()
        question_input = question_info.strip().lower()
        qdrant_answer = self.get_qdrant_response_by_filter(key='author', value=author, question=question_input)
        return self.get_openai_response(qdrant_answer, question)


    def search_by_title(self, question):
        title_info, question_info = question.split('TITLE:', 1)[1].split('INFORMATION:', 1)
        title = title_info.strip().lower()
        question_input = question_info.strip().lower()
        qdrant_answer = self.get_qdrant_response_by_filter(key='title', value=title, question=question_input)
        return self.get_openai_response(qdrant_answer, question)


    def run(self, question):
        return self.agent.run(input=question)


    def _run_in_background(self, question):
        results = self.run(question)
        self.run_in_background_queue.put(results)


    def run_in_background(self, question):
        self.output_buffer = StringIO()
        self.run_in_background_queue = queue.Queue()
        self.run_in_background_thread = threading.Thread(target=self._run_in_background, args=(question,), daemon=True)
        self.run_in_background_thread.start()
        

    def ask_expert_agent(self, question):
        question += f" Answer in {self.language}!"
        self.run_in_background(question=question)
