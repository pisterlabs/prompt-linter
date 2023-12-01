import logging
import constants

import click
import torch
import chromadb
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain import PromptTemplate

from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)
PERSIST_DIRECTORY = f"{constants.ROOT_DIRECTORY}/DB"
client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

def set_qa_prompt():
    """
    Define prompt template for QA
    """
    prompt = PromptTemplate(
        template=constants.QA_TEMPLATE,
        input_variables=["context", "question"],
    )
    return prompt

def load_model(device_type, model_id, model_basename=None):
    """
    This function loads a text-generation model using HuggingFace library.
    If you are running the program for the first time, it will download the model. 
    In subsequent runs, you will use the model from the disk.
    
    Args:
        device_type: device to run on, 'cpu' or 'cuda'
        model_id: Identifier of the model to load from HuggingFace Hub
        model_basename: Basename of the model if using quantized models
            defaults to None.
    
    Returns:
        HuggingFacePipeline: pipeline object that can be used to generate text
    
    Raises:
        ValueError: If provided with unsupported model or device_type
    """
    logging.info(f"Loading model: {model_id}... on {device_type}...")
    logging.info("This action can take a few minutes...")
    
    # Load quantized model
    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for quantized models")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            max_ctx_size = 2048
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)
        else:
            # TODO:
            pass
    # 'model_basename' is None and 'device_type' is cuda
    elif device_type.lower() == 'cuda':
        logging.info("AutoModelForCausalLM for full models...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map='auto',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # Uncomment the line below in case of CUDA out of memory errors
            # max_memory={0: "15GB"}
        )
        logging.info("Model loaded...")
        model.tie_weights()
    # 'model_basename' is None and 'device_type' is cpu
    else:
        logging.info("Using Llama Tokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns
    
    # Create a pipeline for text generation
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("LLM loaded...")
    
    return llm


# Provide command line options on 'device_type'
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(["cpu", "cuda"]),
    help="Device to run on. Defaults to 'cuda' if available, otherwise 'cpu'.",
)
# Provide command line options on 'show_sources'
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources of retrieved documents. Defaults to False.",
)

def main(device_type, show_sources):
    """
    This function runs the following tasks:
    
    1. Load an embedding model. Defaults to HuggingFaceInstructorEmbeddings.
    2. Load a vector store (Chroma) that was created in ingest.py.
    3. Load a LLM using load_model function
    4. Set up a retrieval QA chain
    5. Answer user questions on documents 
    """
    
    logging.info(f"Running on: {device_type}...")
    logging.info(f"Showing sources: {show_sources}...")
    
    # load embedding model
    embeddings = HuggingFaceInstructEmbeddings(model_name=constants.EMBEDDING_MODEL_NAME)
    
    # load vector store
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client=client,
        collection_name="instructor_embeddings",
    )
    # initialize retriever
    retriever = db.as_retriever(search_kwargs={'k': 3})
    
    # load LLM 
    # TODO: 
    model_id = "TheBloke/Llama-2-7B-Chat-GGML"
    model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"
    llm = load_model(device_type, model_id, model_basename)

    # Local CTransformers wrapper for Llama-2-7B-Chat
    # config = {'max_new_tokens': 256,'temperature': 0.01}
    # llm = CTransformers(
    #     model='/Users/minseokbae/ANL/gpt3_finetune/models/llama-2-7b-chat.ggmlv3.q8_0.bin',
    #     model_type='llama',
    #     config=config, 
    # )
    # initialize QA chain
    qa_prompt = set_qa_prompt()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': qa_prompt},
    )
    
    # Interactive questions and answers
    while True:
        query = input("\nAsk a question: ")
        if query == "exit" or query == "quit" or query == "q":
            break
        # Get the answer from the QA chain
        response = qa(query)
        answer, source_documents = response["result"], response["source_documents"]
        
        # Print the answer
        print("\n-> Answer:")
        print(answer)
        
        # Print the source documents if 'show_sources' is True
        if show_sources:
            for document in source_documents:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()