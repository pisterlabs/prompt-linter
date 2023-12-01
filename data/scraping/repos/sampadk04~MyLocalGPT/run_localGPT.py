import os

import torch

from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download

from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, GenerationConfig, pipeline

from constants import CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY, EMBEDDING_MODEL_NAME, MODEL_ID, MODEL_BASENAME, DEVICE_TYPE

def load_model(device_type, model_id, model_basename):
    '''
    Select a model on huggingface.

    Running this for the first time, will download the model and cache it for future runs.

    Args:
        device_type: The device type to use for the model. Either "cpu", "cuda" or "mps"
        model_id: Identifier of the model to load from HuggingFace's model hub
        model_basename: Basename of the model if using quantized models
    
    Returns:
        A HuggingFacePipeline instance
    '''
    # Select the Model ID and model_basename (if need be)
    model_id = model_id
    model_basename = model_basename
    
    if model_basename:
        if device_type.lower() in ['cpu', 'mps']:
            model_path = hf_hub_download(
                repo_id=model_id,
                filename=model_basename
            )

            if device_type.lower() == 'mps':
                return LlamaCpp(
                    model_path=model_path,
                    n_ctx=2048,
                    max_tokens=2048,
                    temperature=0.0,
                    repeat_penalty=1.15,
                    n_gpu_layers=1000
                )
            else:
                return LlamaCpp(
                    model_path=model_path,
                    n_ctx=2048,
                    max_tokens=2048,
                    temperature=0.0,
                    repeat_penalty=1.15
                )

    else:
        # load the tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_id
        )

        # load the model
        model = LlamaForCausalLM.from_pretrained(
            model_id
        )

    # create pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0.0,
        top_p=0.95,
        repetition_penalty=1.15
    )

    # setup the local LLM
    local_llm = HuggingFacePipeline(
        pipeline=pipe
    )

    return local_llm


def main(show_resources):
    # load the instruction embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE_TYPE}
    )

    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )
    retriever = db.as_retriever()

    # set up the prompt template
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    {history}

    Question: {question}
    Helpul Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "history", "question"],
        template=template
    )
    memory = ConversationBufferMemory(input_key="question", memory_key="history")


    # prepare the LLM
    # callbacks = [StreamingStdOutCallbackHandler()]

    # load the LLM for generating Natural Language responses
    llm = load_model(
        device_type=DEVICE_TYPE,
        model_id=MODEL_ID,
        model_basename=MODEL_BASENAME
    )

    # setup langchain pipeline
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt":prompt, "memory":memory}
    )

    # interactive questions and answers loop
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # get answer from the chain
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        # print the query and results
        print("\n\n> Question:")
        print(query)

        print("\n> Answer:")
        print(answer)

        # print the source documents
        if show_resources:
            # this is a flag used to print relevant resources for answers
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")


if __name__ == "__main__":
    main(show_resources=True)