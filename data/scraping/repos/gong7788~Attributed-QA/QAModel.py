import os
from utils import timeit
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

#read api from txt file
api_path = 'api.txt'
prompt_flan = "Please answer the following question.\n"
prompt_openai = "Please answer the following question using the given documents only.\n"

# try:
#     with open(api_path, 'r') as f:
#         lines = f.readlines()
#         lines = [line.strip().split(":")[1] for line in lines]
#         openai_api = lines[0]
#         hf_api = lines[1]
# except FileNotFoundError:
#     print("api.txt file not found")

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api
# os.environ["OPENAI_API_KEY"] = openai_api


def format_prompt_for_flan(question, docs, topk=5) -> str:
    """
    format prompt for flan
    """
    result_docs_list = docs[:topk]
    context = '/n'.join([doc.page_content for doc in result_docs_list])
    
    sample = "Context: {context}\n\nQuestion: {question}\n\nAnswer:".format(context=context, question=question)

    return sample

def answer_from_local_model(question, docs, tokenizer, model, model_name='google/flan-t5-large', ct="stuff", topk=5):
    """
    get answer from a QA model
    
    Args:
        question: str, question to ask
        docs: list of Document, retrieved documents
        model_name: HuggingFace model, DEFAULT: 'google/flan-t5-large'
        ct: str, chain type

    Returns:
        model_answer: str, answer from the model

    """
    # qa_model = HuggingFaceHub(repo_id=model_name)
    # query = prompt_flan + question
    temp = format_prompt_for_flan(question, docs, topk=topk)

    inputs = tokenizer(temp, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs)
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #TODO reminder: results is a list of strings 
    return results[0] 
    # return results

    # chain = load_qa_chain(llm=model, chain_type=ct)
    # model_answer = chain.run(input_documents=docs[:topk], question=query, raw_response=True)
    # print('langchain Results:')
    # print(model_answer)
    # return model_answer
def format_prompt(questions, retrieve_docs):
    formatted_texts = []
    for question, context in zip(questions, retrieve_docs):
        formatted_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        formatted_texts.append(formatted_text)

    return formatted_texts

def local_answer_model(model, tokenizer, questions, retrieve_docs, device, **kwargs):
    temps = format_prompt(questions, retrieve_docs)

    inputs = tokenizer([sentence for sentence in temps], 
                        return_tensors="pt", 
                        padding=True, 
                        max_length=1024, 
                        truncation=True)
    inputs = inputs.to(device)
    
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=512
    )

    model_answers = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

    return model_answers

def answer_from_openai(question, docs, model_name="text-davinci-003", ct="stuff", max_token=1024, topk=5) -> str:
    """
    get answer from a GPT model

    Args:
        question: str, question to ask
        docs: list of Document, retrieved documents
        model_name: str, name of the model to use, DEFAULT: 'text-davinci-003'
        ct: str, chain type
        max_token: int, max token to use, DEFAULT: 1024

    Returns:
        model_answer: str, answer from the model
    """
    query = prompt_openai+question
    model = OpenAI(model_name=model_name, max_tokens=max_token)
    chain = load_qa_chain(llm=model, chain_type=ct)
    model_answer = chain.run(input_documents=docs[:topk], question=query, raw_response=True)
    return model_answer