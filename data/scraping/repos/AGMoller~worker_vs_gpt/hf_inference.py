import json
import os
from langchain import HuggingFaceTextGenInference, HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
import requests
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from worker_vs_gpt.config import EMPATHY_DATA_DIR, SAMESIDE_DATA_DIR

MODEL = "meta-llama/Llama-2-7b-chat-hf"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
headers = {"Authorization": "Bearer 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"}


load_dotenv()


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = query(
    {
        "inputs": """<s>[INST] <<SYS>>
You are an AI assistant. Your job is to answer questions about Anders, and only about Anders. He is a PhD student at IT University of Copenhagen. Answer like you're a pirate.
<</SYS>>

Who is Anders? Please explain in great detail: [/INST]""",
        "parameters": {
            "options": {"wait_for_model": True},
            "temperature": 0.7,
            "min_length": 1000,
            "max_length": 4000,
            "top_k": 9,
            # "top_p": 0.7,
            "do_sample": True,
        },
    }
)

system_message = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=[],
        template="""
        You are an advanced classifying AI. You are tasked with classifying the whether the text expresses empathy.
        """,
    )
)

human_message = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["few_shot", "text"],
        template="""Based on the following text, classify whether the text expresses empathy or not. You answer MUST only be one of the two labels. Your answer MUST be exactly one of ['empathy', 'not empathy']. The answer must be lowercased.
{few_shot}
Text: {text}

Answer:
""",
    )
)
prompt = ChatPromptTemplate.from_messages([system_message, human_message])

llm = HuggingFaceHub(
    repo_id=MODEL,
    task="text-generation",
)

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


chain = LLMChain(
    prompt=prompt,
    llm=llm,
    verbose=True,
)

train = pd.read_json(os.path.join(EMPATHY_DATA_DIR, "train.json"))

from worker_vs_gpt.utils import few_shot_sampling

few_shot_samples = few_shot_sampling(df=train, n=0)

print(API_URL)

print(
    chain.run(
        {
            "few_shot": few_shot_samples,
            "text": "Poor sad polar bear! They need to move him now. If they are going to take polar bears and other animals out of their natural habitats, then they better make sure that the place they keep them in is at the right standard for those animals. They need to move that polar bear as soon as possible so that it has a chance at happiness.",
        },
        callbacks=[StreamingStdOutCallbackHandler()],
    ),
)
