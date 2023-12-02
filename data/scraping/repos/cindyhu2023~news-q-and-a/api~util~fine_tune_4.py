from haystack.document_stores import OpenSearchDocumentStore
from haystack.nodes import  PromptTemplate, EmbeddingRetriever
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
from config import long_prompt_template, short_prompt_template, default_prompt_template, temperature_prompt_template, questions
from utils import remove_duplicate_references

load_dotenv()

url = os.getenv("OPENSEARCH_URL")
username =  os.getenv("OPENSEARCH_USERNAME")
password = os.getenv("OPENSEARCH_PASSWORD")
document_store = OpenSearchDocumentStore(
    host=url, username=username, password=password, 
    port=443, verify_certs=True,
)

retriever = EmbeddingRetriever(
    document_store=document_store,
   embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
   model_format="sentence_transformers",
   top_k=5
)

client = OpenAI(
        api_key=os.getenv("OPEN_AI_KEY")
    )
def query(user_query, temperature, prompt_template):
    documents = retriever.retrieve(query=user_query)
    prompt_template_obj = PromptTemplate(prompt=prompt_template)
    filled_prompt = list(prompt_template_obj.fill(documents=documents, query=user_query))[0]
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": filled_prompt,
            }
        ],
        model="gpt-3.5-turbo",
        max_tokens=1024,
        temperature=temperature,
    )

    answer = chat_completion.choices[0].message.content
    reference = {}
    for idx, doc in enumerate(documents):
        reference[idx+1] = doc.meta["URL"]
    
    answer, reference = remove_duplicate_references(answer, reference)

    return answer, reference

def run():
    # logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.DEBUG)
    # logging.getLogger("haystack").setLevel(logging.INFO)
    responses1 = {}
    templates = [
        {"name": "default", "template": default_prompt_template},
        {"name": "short", "template": short_prompt_template},
        {"name": "long", "template": long_prompt_template}
    ]
    for template in templates:
        res = []
        for question in questions:
            answer, reference = query(question, 0.5, template["template"])
            res.append({
                "question": question,
                "answer": answer,
                "reference": reference
            })
        responses1[template["name"]] = res
    with open("sample_response_1.5.json", 'w') as outfile:
          json.dump(responses1, outfile)
    print("Done writing sample_response_1.json")

    # responses2 = {}
    # temps = [
    #     {"name": "temperature_0.5", "temperature": 0.5},
    #     {"name": "temperature_0.9", "temperature": 0.9},
    #     {"name": "temperature_0.1", "temperature": 0.1}
    # ]
    # for temp in temps:
    #     print("temp: ", temp["name"])
    #     res = []
    #     for question in questions:
    #         print("question: ", question)
    #         answer, reference = query(question, temp["temperature"], temperature_prompt_template)
    #         res.append({
    #             "question": question,
    #             "answer": answer,
    #             "reference": reference
    #         })
    #     responses2[temp["name"]] = res
    # with open("sample_response_6.json", 'w') as outfile:
    #       json.dump(responses2, outfile)

    return

run()

