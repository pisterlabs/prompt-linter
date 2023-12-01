from glob import glob
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
import os, sys, openai
from dotenv import load_dotenv
from langchain.tools import tool, Tool
from retrieval import Retrieval
from pyepsilla import cloud

load_dotenv() 

retrieval = Retrieval()

client = cloud.Client(
  project_id=os.getenv("PROJECT_ID"),
  api_key=os.getenv("EPSILLA_API_KEY")
)
db = client.vectordb(db_id=os.getenv("DB_ID"))

@tool
def search_api(question: str) -> str:
    """Searches the relevant information from the document set to answer the question."""
    qs = retrieval.rephrase(question=question)
    query_score_dict = {}
    item = retrieval.vector_search(db, question)
    # print(item)
    query_score_dict[question] = item
    for q in qs:
      item = retrieval.vector_search(db, q)
      query_score_dict[q] = item
    # print(query_score_dict)

    ranking_result = retrieval.ranking_fusion(original_query=question, query_score_dict=query_score_dict)
    final_result = retrieval.generate_content_based_on_ranking(ranking_result)
    return final_result

class DocAgent:
    def __init__(self):
        api_key = os.getenv("OPENAI_KEY")
        llm = OpenAI(temperature=0, openai_api_key=api_key)
        tools = [search_api]
        tools = tools + load_tools(["llm-math"], llm=llm)
        self.agent_executor = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )
    
    def list_docs(self):
        # List PDF files under ./documents/ folder
        ret = []
        files = glob("./documents/*.pdf")
        for pdf in files:
            ret.append(os.path.basename(pdf))
        return ret
  

    def solve(self, question):
        response = self.agent_executor.invoke(
            {
                "input": question
            }
        )
        return response['output']
    
    def rephrase(self, question):
        return retrieval.rephrase(question=question)
    
    def solve_one(self, file, question, questions):
        # Step 1. Search the relevant information from the document to answer the question.
        query_score_dict = {}
        item = retrieval.vector_search_one_doc(db, question, file)
        # print(item)
        query_score_dict[question] = item
        for q in questions:
          item = retrieval.vector_search_one_doc(db, q, file)
          query_score_dict[q] = item
        # print(query_score_dict)

        ranking_result = retrieval.ranking_fusion(original_query=question, query_score_dict=query_score_dict)
        context = retrieval.generate_content_based_on_ranking(ranking_result)

        # Step 2. Use the prompt to answer the question for the document.
        openai.api_key = os.getenv("OPENAI_KEY")
        response = openai.ChatCompletion.create(
          model="gpt-4",
          messages=[
            {
              "role": "system",
              "content": "You are an assistant answering questions for a given document."
            },
            {
              "role": "user",
              "content": f'''
                Answer the Question based on the given Context. Please don't make things up. Ask for more information when needed.

                Context:
                {context}

                Question:
                {question}

                Answer:
                Let's work this out in a step by step way to be sure we have the right answer.
                '''
            }
          ],
          temperature=0,
          max_tokens=256,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )

        return response['choices'][0]['message']['content']

    def summary(self, question, concated):
        openai.api_key = os.getenv("OPENAI_KEY")
        response = openai.ChatCompletion.create(
          model="gpt-4",
          messages=[
            {
              "role": "system",
              "content": "You are an assistant answering questions for a given document."
            },
            {
              "role": "user",
              "content": f'''
                Answer the Question based on the Analysis Of Each Document. If some documents are not related to the question, please ignore them.

                Analysis Of Each Document:
                {concated}

                Question:
                {question}

                Answer:
                '''
            }
          ],
          temperature=0,
          max_tokens=256,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )

        return response['choices'][0]['message']['content']
    
    def can_loop(self, question):
        openai.api_key = os.getenv("OPENAI_KEY")
        response = openai.ChatCompletion.create(
          model="gpt-4",
          messages=[
            {
              "role": "system",
              "content": "You are an assistant answering questions for a large set of documents."
            },
            {
              "role": "user",
              "content": "For the provided question, determine if we can check the documents one by one and make the judgement and answer it purely based on facts from this file, or we need to cross validate with other files. If former, response \"YES\"; if later, response \"NO\"\n\nQuestion: " + question
            }
          ],
          temperature=0,
          max_tokens=256,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
        decision = response['choices'][0]['message']['content'] or 'YES'
        return decision == 'YES'


# agent = DocAgent()
# print(agent.list_docs())



