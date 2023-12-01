import os
import chromadb
import openai
from text_helpers import document_info
from llm_interface import LLMInterface, get_llm_interface

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

with open('prompts/system_prompt.txt', 'r') as f:
    SYSTEM_PROMPT = f.read()
with open('prompts/user_prompt.txt', 'r') as f:
    USER_PROMPT = f.read()
with open('prompts/query_system_prompt.txt', 'r') as f:
    QUERY_SYSTEM_PROMPT = f.read()

client = chromadb.PersistentClient(path="db/")
print(f"{client.list_collections()}")

def retrieve(question):
    print(f'initial question: {question}')
    # query_text = get_chat_completion(QUERY_SYSTEM_PROMPT, question)
    # print(f'query text: {query_text}')
    res={}
    for document_type in ['shot', 'run', 'miniproposal']:
        n_results=document_info[document_type]['n_documents']
        if n_results>0:
            collection = client.get_collection(f'{document_type}_embeddings')
            qr = collection.query(query_texts=question, n_results=n_results)
            ids = qr['ids'][0]
            documents = qr['documents'][0]
            # change this into a dict or something
            res.update({k: v for k, v in zip(ids, documents)})
    return res

def process_results(results):
    processed_results = f""
    for k, v in results.items():
        processed_results = processed_results + f"{k}: {v}\n"
    return processed_results

def rag_answer_question(question, results, model: LLMInterface):
    processed_results = process_results(results)
    formatted_user_prompt = USER_PROMPT.format(question=question, results=processed_results)
    print(formatted_user_prompt)
    return model.query(SYSTEM_PROMPT, formatted_user_prompt)

def test():
    question = "Tell me about shots that struggled with tearing modes"
    model = get_llm_interface("openai")
    results = retrieve(question)
    answer = rag_answer_question(question, results, model)
    print(f"Model {model.model_name} answer:\n{answer}")

if __name__ == '__main__':
    test()
