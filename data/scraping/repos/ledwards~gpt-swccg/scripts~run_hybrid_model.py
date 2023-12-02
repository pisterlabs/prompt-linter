import os
from dotenv import load_dotenv
import openai

load_dotenv()

openai.organization = "org-D2FBgBhwLFkKAOsgtSp86b4i"
openai.api_key = os.getenv("OPENAI_API_KEY")

remote_files = openai.File.list()["data"]
training_files = filter(lambda f: "training.jsonl" in f["filename"], remote_files)
latest_file_id = max(training_files, key=lambda x: x["created_at"])["id"]

fine_tunes = openai.FineTune.list()["data"]
latest_fine_tune_model = max(fine_tunes, key=lambda x: x["created_at"])["fine_tuned_model"]

def create_context(question, max_len=1800, search_model='ada', max_rerank=10):
    results = openai.Engine(search_model).search(
        search_model=search_model,
        query=question,
        max_rerank=max_rerank,
        file=latest_file_id,
        return_metadata=True
    )
    returns = []
    cur_len = 0

    for result in results['data']:
        cur_len += len(result['metadata']) + 4
        if cur_len > max_len:
            break
        returns.append(result['text'])
    return "\n\n###\n\n".join(returns)

def answer_question(question, fine_tuned_qa_model, max_len=1800, search_model='ada', max_rerank=10, debug=False):
    context = create_context(question, max_len=max_len, search_model=search_model, max_rerank=max_rerank)
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    try:
        response = openai.Completion.create(
            model=fine_tuned_qa_model,
            prompt=f"Answer the question based on the context below\n\nText: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response['choices'][0]['text']
    except Exception as e:
        print (f'ERROR: {e}')
        print(f'Latest fine-tune model: {latest_fine_tune_model}')
        return ""

question = "Who called Luke Skywalker 'Wormie'?"
print(question)

answer = answer_question(question, latest_fine_tune_model)
print(answer)