import openai
import cohere
import os
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import models
from dotenv import load_dotenv

load_dotenv()

collection_name = "hackathon_collection"

# Title: The Will to Believe and Other Essays in Popular Philosophy
# Author: William James

def cohere_embedding(question):
    cohere_client = cohere.Client(os.environ.get('COHERE_API_KEY'))
    model = 'multilingual-22-12'
    response = cohere_client.embed(
        texts=[question],
        model=model,
    )
    embedding = [float(e) for e in response.embeddings[0]]
    return embedding

def openai_response(question, qdrant_answer):
    prompt = ""
    for r in qdrant_answer:
        prompt += f"""excerpt: author: {r.payload.get('author')}, title: {r.payload.get('title')}, text: {r.payload.get('text')}\n"""
    
    # # TODO - figure out a relevant limit for contextual information
    # if len(prompt) > 10000:
    #     prompt = prompt[0:10000]

    prompt += f"""
Given the excerpts above, answer the following question:
Question: {question}"""
#     prompt = f"""
#         Given the texts below, answer the following question:
#         Question: {question}

#         Texts:
#         """
#     for answer in answers_list:
#         prompt += '{}\n'.format(answer)
    
    messages = [{"role": "user", "content": prompt}]
    
    openai_model = 'gpt-3.5-turbo'
    response = openai.ChatCompletion.create(
        model=openai_model,
        messages=messages,
        temperature=0.1,
        max_tokens=1000,
        # frequency_penalty=0.0,
        # presence_penalty=0.0,
        # stop=["\n"]
    )

    # open_ai_response.append(response.choices[0].message.content)
    print(prompt)
    return response.choices[0].message.content
    

def qdrant_search_by_filter(key, value, question):
    # perform author payload filter + information vector search
    embedding = cohere_embedding(question)
    db_client = QdrantClient(
        api_key=os.environ.get('QDRANT_API_KEY'),
        host=os.environ.get('QDRANT_HOST')
    )
    response = db_client.search(
        collection_name=collection_name,
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
        query_vector=embedding,
        limit=5
    )

    return response

def search_author(input):
    author_info, question_info = input.split('AUTHOR:', 1)[1].split('INFORMATION:', 1)
    author = author_info.strip().lower()
    question = question_info.strip().lower()
    qdrant_answer = qdrant_search_by_filter(key='author', value=author, question=question)
    # answers_list = []
    # for count, answer in enumerate(qdrant_answer):
    #     if count == 0:
    #         author_name = answer.payload.get('author', 'unknown')
    #         answers_list.append(f"author name is {author_name}")    
    #     answers_list.append(f"{answer.payload.get('text')}")
    
    return openai_response(question, qdrant_answer)


def search_title(input):
    # perform title payload filter + information vector search
    title_info, question_info = input.split('TITLE:', 1)[1].split('INFORMATION:', 1)
    title = title_info.strip().lower()
    question = question_info.strip().lower()
    qdrant_answer = qdrant_search_by_filter(key='title', value=title, question=question)
    # answers_list = []
    # for count, answer in enumerate(qdrant_answer):
    #     if count == 0:
    #         title = answer.payload.get('title', 'unknown')
    #         answers_list.append(f"title is {title}")
    #     answers_list.append(f"{answer.payload.get('text')}")
    
    return openai_response(question, qdrant_answer)
        
    

def qdrant_search(question):
    embedding = cohere_embedding(question)

    db_client = QdrantClient(
        api_key=os.environ.get('QDRANT_API_KEY'),
        host=os.environ.get('QDRANT_HOST')
    )
    qdrant_answer = db_client.search(
        collection_name="hackaton_collection",
        query_vector=embedding,
        limit=8,
    )
    print(f"QDRANT ANSWER TYPE {type(qdrant_answer)}")
    print(f"QDRANT ANSWER {qdrant_answer}")


    # answers_list = []
    # for answer in qdrant_answer:
    #     answers_list.append('{}\n'.format(answer.payload.get('text')))

    return openai_response(question, qdrant_answer)

tools = [
    Tool(
        name="search_internal_knowledge_base",
        func=lambda question: qdrant_search(question),
        description="""Useful for searcing the internal knowledge base about general.
Only use this tool if no other specific search tool is suitable for the task."""
        # description="use when searching for information filtering by a specific author.",
        # description="use when you want to discover who is the author, asking a question with informations you have",
    ),
    Tool(
        name="search_internal_knowledge_base_for_specific_author",
        func=lambda x: search_author(x),
        description="""Only use this tool when the name of the specific author is known and mentioned in the question.
Use this tool for searching information about this specific author.
If the name of the author is not explicitly mentioned in the original question DO NOT USE THIS TOOL.
The input to this tool should contain the name of the author and the information you are trying to find. 
Input template: 'AUTHOR: name of the author INFORMATION: the information you are searching for in the form of a long and well composed question'"""
        # description="use when you know the author's name and want to filter results based on their name and other informations that you have. create input like 'author: information:'"
        # description="use when searching for information filtering by a specific author.",
        # description="use when you want to discover who is the author, asking a question with informations you have",
    ),
    Tool(
        name="search_internal_knowledge_base_for_specific_document_title",
        func=lambda x: search_title(x),
        description="""Use this only when you are searching for information about one specific document title 
and you know this document's title. Do not use this if you do not know the document's title. 
Create an input with the title of the document and the information you are searching for them.
Input template: 'TITLE: title of the document INFORMATION: the information you are searching for in the form of a long and well composed question'"""
        # description="use when searching for information filtering by a specific title.",
        # description="use when you want to discover which is the title, asking a quesiton with informations you have",
    )
]

agent = initialize_agent(
    tools=tools, 
    llm=OpenAI(temperature=0.1), 
    agent="zero-shot-react-description", 
    verbose=True,
    # return_intermediate_steps=True
)

if __name__ == '__main__':
    question = 'who wrote about his posthumous memories?'
    # question = 'quem é o autor que escreve sobre suas memórias póstumas?'
    # question = 'em qual trecho o machado de assis comenta sobre filhos?'
    # question = 'compare o que esses autores disseram sobre a vida em sociedade: Machado de Assis, Henry David Thoreau, Yuval Noah Harari'
    # question = 'sobre quais épocas cada um desses autores escreve: Machado de Assis, Henry David Thoreau, Yuval Noah Harari'
    # question = 'a obra de cada um desses autores se passa em uma determinada época, tendo um contexto histórico daquela época em específico. Em qual contexto histórico a obra de cada um desses autores se encaixa, ou seja, em qual época se passa a obra de cada um desses autores? Escritores: Machado de Assis, Henry David Thoreau, Yuval Noah Harari'
    # question = 'as obras desse autor se passam em diferentes épocas, tendo um contexto histórico espcífico sobre cada época em específico. Em qual contexto histórico as obras desse autor se encaixa, ou seja, em qual época se passa a obra de cada um desses autores? Escritores: Machado de Assis, Henry David Thoreau, Yuval Noah Harari'
    # question = 'o que há de comum nas obras desses três autores: Machado de Assis, Henry David Thoreau, Yuval Noah Harari'
    # question = 'compare o que disseram Machado de Assis e Henry David Thoreau sobre a sociedade?'
    # question = 'qual é o titulo do livro em o autor machado de assis comenta sobre filhos?'
    # question = 'Com efeito, um dia de manhã, estando a passear na chácara, pendurou-se-me uma idéia no trapézio que eu tinha no cérebro.']

    response = agent.run(input=question)
    print("RESPONSE DO AGENT")
    print(response)