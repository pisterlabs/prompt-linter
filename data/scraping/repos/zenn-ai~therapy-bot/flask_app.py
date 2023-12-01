import json
import os
import warnings

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftConfig, PeftModel

import sys
sys.path.append("../inference")
from zen_intent import zen_intent_classifer, prompt_from_intent

from fastchat.utils import get_gpu_memory, is_partial_stop, is_sentence_complete, get_context_length
from fastchat.conversation import get_conv_template, register_conv_template, Conversation, SeparatorStyle
from fastchat.serve.inference import generate_stream

from flask import Flask, jsonify, request
import streamlit as st

import firebase as pyrebase
from datetime import datetime

from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings

warnings.filterwarnings('ignore')

SYSTEM_MSG = """Your name is Zen and you're a mental health counselor. Please have a conversation with your patient and provide them with a helpful response to their concerns."""

# Configuration Key
firebaseConfig = {
    'apiKey': "AIzaSyBvAeBh-ghFe-4n9VSNTSW_h9zCT3bXngg",
    'authDomain': "cloud-lab-ff59.firebaseapp.com",
    'projectId': "cloud-lab-ff59",
    'databaseURL': "https://cloud-lab-ff59-default-rtdb.firebaseio.com/",
    'storageBucket': "cloud-lab-ff59.appspot.com",
    'messagingSenderId': "1016159354336",
    'appId': "1:1016159354336:web:862ea0d538eee01c11ff85",
}
# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()


try:
    register_conv_template(
        Conversation(
            name="Zen",
            system_message=SYSTEM_MSG,
            roles=("USER", "ASSISTANT"),
            sep_style=SeparatorStyle.ADD_COLON_TWO,
            sep=" ",
            sep2="</s>",
        )
    )
except AssertionError:
    pass


def get_embeddings():
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs=encode_kwargs
    )
    return embeddings


def create_vectorstore(documents):
    embeddings = get_embeddings()

    qdrant = Qdrant.from_documents(
        documents, embeddings, 
        location=":memory:",
        collection_name="convo"
    )

    return qdrant


# Function to send message to Firebase
def send_message(user_id, conv):
    for sender, message in conv:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S.%f")
        message_data = {'message': message, 'timestamp': dt_string, 'sender': sender}
        db.child(user_id).child("Messages").push(message_data)


# Fetch Conversation History from Firebase and sort in descending order of timestamp
def get_chat_history(user_id):
    messages = db.child(user_id).child("Messages").get()
    chat_history = [{'message': message.val()['message'], 'timestamp': message.val()['timestamp'], 'sender': message.val()['sender']} for message in messages.each()] if messages.val() else []
    chat_history = sorted(chat_history, key=lambda x: datetime.strptime(x['timestamp'], "%d/%m/%Y %H:%M:%S.%f"), reverse=False)
    chat_history = [[x["sender"].upper(), x["message"]] for x in chat_history]
    
    conv = get_conv_template("Zen")
    conv.set_system_message(SYSTEM_MSG)
    conv.messages = chat_history
    return conv


def load_model(model_path, num_gpus, max_gpu_memory=None):
    
    kwargs = {"torch_dtype": torch.float16}
    if num_gpus != 1:
        kwargs["device_map"] = "auto"
        if max_gpu_memory is None:
            kwargs[
                "device_map"
            ] = "sequential"  # This is important for not the same VRAM sizes
            available_gpu_memory = get_gpu_memory(num_gpus)
            kwargs["max_memory"] = {
                i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                for i in range(num_gpus)
            }
        else:
            kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
        
        config = PeftConfig.from_pretrained(model_path)
        base_model_path = config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, use_fast=False
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        
        return model, tokenizer, base_model


use_vicuna = False
num_gpus = 4
max_gpu_memory = "12GiB"
model_path = "/home/jupyter/therapy-bot/models/ZenAI/"
if use_vicuna:
    _, tokenizer, model = load_model(model_path, num_gpus, max_gpu_memory)
else:
    model, tokenizer, _ = load_model(model_path, num_gpus, max_gpu_memory)


def get_context(conv, question):
    messages = [x[0] + ": " + x[1] for x in conv.messages]
    documents = ['\n'.join(messages[i:i + 10]) for i in range(0, len(messages), 10)]
    
    vectordb = create_vectorstore(documents)
    context = vectordb.similarity_search_with_score(question, k=1)
    return context


def chat_streamlit(
    model, tokenizer, user_id, question,
    device, num_gpus, max_gpu_memory,
    conv_template="Zen", system_msg=SYSTEM_MSG,
    temperature=0.7, repetition_penalty=1.0, max_new_tokens=512,
    dtype=torch.float16,
    judge_sent_end=True
):
    context_len = get_context_length(model.config)
    
    conv = get_chat_history(user_id)
    
    intent = zen_intent_classifer(question)
    if intent == 4:
        system_message = prompt_from_intent(intent)
        return conv, system_message
    
    if intent == 2 or intent == 3:
        system_message = prompt_from_intent(intent)
        conv.set_system_message(system_message)
    
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    if intent == 1 and len(tokenizer.encode(prompt)) > context_len - 500:
        system_message = prompt_from_intent(intent)
        context = get_context(question)
        system_message += context + "\n\n"

        conv.set_system_message(system_message)
        prompt = conv.get_prompt()
        
    gen_params = {
        "prompt": prompt,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }
    print("---", prompt)

    output_stream = generate_stream(
        model,
        tokenizer,
        gen_params,
        device,
        context_len=context_len,
        judge_sent_end=judge_sent_end,
    )
    
    return conv, output_stream


def stream_output(output_stream):
    pre = 0
    for outputs in output_stream:
        output_text = outputs["text"]
        output_text = output_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            # print(" ".join(output_text[pre:now]), end=" ")
            pre = now
    # print(" ".join(output_text[pre:]))
    return " ".join(output_text)


app = Flask(__name__)

def get_answer():
    question = request.json.get("question", "")
    user_id = request.json.get("user_id", "")
    print("-----------------------------------------", question, "------------------------------------------")
    conv, output_stream = chat_streamlit(
        model=model,
        tokenizer=tokenizer,
        user_id=user_id,
        question=question,
        device="cuda",
        num_gpus=4,
        max_gpu_memory="12GiB"
    )
    if isinstance(output_stream, str):
        answer = output_stream
    else:
        answer = stream_output(output_stream).strip()
    conv.update_last_message(answer)
    
    print("##############################################################################")
    print(conv.get_prompt())
    print("##############################################################################")
    
    send_message(user_id, conv.messages[-2:])
    return answer
    

@app.route('/api/data', methods=["POST"])
def get_data():
    answer = get_answer()
    return jsonify({"answer": answer}), 200, {"Content-Type": "application/json"}


if __name__ == "__main__":
    app.run(host="localhost", port="5000", debug=False)