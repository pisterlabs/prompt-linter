import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.llms import HuggingFaceLLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context
from llama_index import ServiceContext
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (
    download_loader,
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.prompts import Prompt
from llama_index.chat_engine.condense_question import CondenseQuestionChatEngine
from llama_index.memory import ChatMemoryBuffer
from llama_index.llms import ChatMessage, MessageRole
# from .text_to_speech import TextToSpeech
import shutil
import random
import datetime


class AIModel:
    def __init__(self, model_name, model_path, quantization, api_key):
        if quantization:
            model_url = None
            if "http" in model_path:
                model_url = model_path
                model_path = None
            else:
                model_path = model_path

            print("Loading model...")
            print(model_path)

            # Create a llama2 model
            llm = LlamaCPP(
                # You can pass in the URL to a GGML model to download it automatically
                model_url=model_url,
                # optionally, you can set the path to a pre-downloaded model instead of model_url
                model_path=model_path,
                temperature=0.1,
                max_new_tokens=512,
                # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
                context_window=3900,
                # kwargs to pass to __call__()
                generate_kwargs={},
                # kwargs to pass to __init__()
                # set to at least 1 to use GPU
                model_kwargs={"n_gpu_layers": 15},
                # transform inputs into Llama2 format
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                verbose=True,
            )
        else:
            # Create a HF LLM using the llama index wrapper
            self.tokenizer = (
                AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir="./model/",
                    user_auth_token=api_key,
                ),
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir="./model/",
                use_auth_token=api_key,
                load_in_4bit=True,
            )

            # Create a HF LLM using the llama index wrapper
            llm = HuggingFaceLLM(
                context_window=4096,
                max_new_tokens=256,
                system_prompt=system_prompt,
                query_wrapper_prompt=query_wrapper_prompt,
                model=self.model,
                tokenizer=tokenizer,
            )

        # Create and dl embeddings instance
        self.embeddings = LangchainEmbedding(
            HuggingFaceEmbeddings(
                model_name="all-MiniLM-L12-v2",
                model_kwargs= {
                    "device": "cuda:0" if torch.cuda.is_available() else "cpu"
                }
            )
            # HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )

        # Bring in stuff to change service context
        self.service_context = ServiceContext.from_defaults(
            # chunk_size=1024,
            llm=llm,
            embed_model=self.embeddings,
        )

        set_global_service_context(self.service_context)

        self.train()

        # self.tts = TextToSpeech()

        self.continue_count = 0

    # Trains the AI model
    # if [hard] is true, then all the vectors indices will be purged
    # and be built from scratch
    def train(self, hard=False):
        # Train the AI model
        folder = "data/training_data"

        if hard and os.path.exists("data/indices"):
            shutil.rmtree("data/indices")


        if os.path.exists("data/indices"):
            storage_context = StorageContext.from_defaults(persist_dir="data/indices")
            self.index = load_index_from_storage(storage_context)
        else:
            file_metadata = lambda x: {"filename": x}
            data = SimpleDirectoryReader(
                input_dir=folder, exclude=["*.png"], recursive=True,
                file_metadata=file_metadata
            ).load_data()
            self.index = VectorStoreIndex.from_documents(data)
            self.index.storage_context.persist("data/indices")
        # Prompt template: Llama-2-Chat
        self.system_prompt = """
[INST]
<<SYS>>
You are Iron Man's JARVIS.
Your Job is to assist Mr. 'Praharsh Bhatt' with his queries.
<</SYS>>
- At the end of your response, add '/end' to end the conversation.
- When you get the prompt '/continue', you continue your response from the last message, without any text preceding it.
[/INST]
"""

        self.chat_engine = self.index.as_chat_engine(
            service_context=self.service_context,
            verbose=True,
            memory=ChatMemoryBuffer.from_defaults(
            ),
            chat_mode="context",
            system_prompt=self.system_prompt,
        )

    def query(self, query, should_speak=False):

        # Get a response from the Model
        starting_datetime = datetime.datetime.now()
        response = self.chat_engine.chat(query).response.strip()
        # response = "test /end"
        ending_datetime = datetime.datetime.now()
        print("Time taken: ", (ending_datetime - starting_datetime))
 
        # Format the response
        print("Response: ", response)
        if len(response) == 0:
            response = (
                "Sorry, I didn't get that. Could you please rephrase your question?/end"
            )

        # If the response ends with '/end', remove it
        # if "/end" in response:
        #     response = response.replace("/end", "")
        # else:
        #     if (
        #         self.continue_count < 4
        #     ):  # 4 is the max number of times the model can continue
        #         self.continue_count += 1
        #         response = response + self.query("/continue")

        # if should_speak:
        #     self.tts.speak(response)
        return response

    # Reset the chat engine
    def reset(self):
        self.chat_engine.reset()
