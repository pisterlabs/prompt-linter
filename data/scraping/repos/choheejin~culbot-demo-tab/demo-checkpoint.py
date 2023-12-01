import torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel, PeftConfig
import fire
import pandas as pd

from utils.prompter import Prompter
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import openai

import gradio as gr
import time

def query_collection(collection, query, max_results, dataframe):
    results = collection.query(query_texts=query, n_results=max_results, include=['distances']) 
    df = pd.DataFrame({
                'id':results['ids'][0], 
                'score':results['distances'][0],
                'question': dataframe[dataframe.vector_id.isin(results['ids'][0])]['question'],
                'answer': dataframe[dataframe.vector_id.isin(results['ids'][0])]['answer'],
                })
    return df

def respond1(
    message,
    chat_history,
):
    def gen(instruction="", input_text=""):
        gc.collect()
        torch.cuda.empty_cache()
        prompt = prompter.generate_prompt(instruction=instruction, input=input_text)
        output = pipe(prompt, max_length=1024, temperature=0.2, num_beams=5, eos_token_id=2)
        s = output[0]["generated_text"]
        result = prompter.get_response(s)
        return result
        
    query_result = query_collection(
        collection=cbnu_question_collection,
        query=message,
        max_results=3,
        dataframe=df2
    )
    qa_set = ''
    question_list = query_result.question
    answer_list = query_result.answer
    for q, a  in zip(question_list,answer_list):
        qa_set += "question: {} answer: {} /n".format(q,a)
    print(qa_set)

    bot_message = gen(instruction=message, input_text=qa_set)
    print(bot_message)
    chat_history.append((message, bot_message))
    time.sleep(0.5)

    return "", chat_history

def respond2(
    message,
    chat_history,
):
    def gen(instruction="", input_text=""):
        gc.collect()
        torch.cuda.empty_cache()
        prompt = prompter.generate_prompt(instruction=instruction, input=input_text)
        output = pipe(prompt, max_length=1024, temperature=0.2, num_beams=5, eos_token_id=2)
        s = output[0]["generated_text"]
        result = prompter.get_response(s)
        return result
    
    query_result = query_collection(
            collection=medical_question_collection,
            query=message,
            max_results=3,
            dataframe=df
        )
    qa_set = ''
    question_list = query_result.question
    answer_list = query_result.answer
    for q, a  in zip(question_list,answer_list):
        qa_set += "question: {} answer: {} /n".format(q,a)
    print(qa_set)

    bot_message = gen(instruction=message, input_text=qa_set)
    print(bot_message)
    chat_history.append((message, bot_message))
    time.sleep(0.5)

    return "", chat_history


with gr.Blocks(title="Culbot") as demo:
    gr.Markdown("### Culbot")
    with gr.Tab("충북대 관련 질문"):
        chatbot = gr.Chatbot().style(height=550)
        with gr.Row():
            with gr.Column(scale=9):
                # 입력
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter",
                ).style(container=False)
            with gr.Column(scale=1):
                # 버튼
                clear = gr.Button("➤")
        # 버튼 클릭
        clear.click(respond1, [msg, chatbot], [msg, chatbot])
        # 엔터키
        msg.submit(respond1, [msg, chatbot], [msg,chatbot])
    
    with gr.Tab("질병 관련 질문"):
        chatbot = gr.Chatbot().style(height=550)
        with gr.Row():
            with gr.Column(scale=9):
                # 입력
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter",
                ).style(container=False)
            with gr.Column(scale=1):
                # 버튼
                clear = gr.Button("➤")
        # 버튼 클릭
        clear.click(respond2, [msg, chatbot], [msg, chatbot], "medical")
        # 엔터키
        msg.submit(respond2, [msg, chatbot], [msg,chatbot], "medical")    

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    chroma_client = chromadb.Client()

    df = pd.read_json('./data_medical_embedding2.json')
    df.rename(columns={'instruction': 'question', 'output':'answer'}, inplace=True)

    df2 = pd.read_json('./data_cbnu_embedding2.json')
    df2.rename(columns={'instruction': 'question', 'output':'answer'}, inplace=True)

    OPENAI_API_KEY = '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'
    openai.api_key = OPENAI_API_KEY
    
    embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name='text-embedding-ada-002')
    
    medical_question_collection = chroma_client.create_collection(name='medical_question', embedding_function=embedding_function)
    medical_answer_collection = chroma_client.create_collection(name='medical_answer', embedding_function=embedding_function)

    cbnu_question_collection = chroma_client.create_collection(name='cbnu_question', embedding_function=embedding_function)
    cbnu_answer_collection = chroma_client.create_collection(name='cbnu_answer', embedding_function=embedding_function)

    df = df.astype({'vector_id':'str'})
    df2 = df2.astype({'vector_id':'str'})

    batch_size = 166
    chunks = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
    chunks2 = [df2[i:i + batch_size] for i in range(0, len(df2), batch_size)]

    # Add the content vectors in batches
    for chunk in chunks:
        medical_question_collection.add(
            ids=chunk['vector_id'].tolist(),
            embeddings=chunk['question_vector'].tolist(),  # Assuming you have the 'question_vector' column
        )
        medical_answer_collection.add(
            ids=chunk['vector_id'].tolist(),
            embeddings=chunk['answer_vector'].tolist(),  # Assuming you have the 'answer_vector' column
        )

    for chunk in chunks2:
        cbnu_question_collection.add(
            ids=chunk['vector_id'].tolist(),
            embeddings=chunk['question_vector'].tolist(),  # Assuming you have the 'question_vector' column
        )
        cbnu_answer_collection.add(
            ids=chunk['vector_id'].tolist(),
            embeddings=chunk['question_vector'].tolist(),  # Assuming you have the 'question_vector' column
        )
        

    MODEL = "EleutherAI/polyglot-ko-12.8b"
    LORA_WEIGHTS = "yeongsang2/polyglot-ko-12.8B-v.1.02-checkpoint-4500"

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map={"":0})
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
    model.eval()

    pipe = pipeline("text-generation", model=model, tokenizer=MODEL)
    prompter = Prompter("cbnu")

    demo.launch(server_name="0.0.0.0", server_port=5000)