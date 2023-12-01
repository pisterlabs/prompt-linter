import os

# render UI
import gradio
# for LLMs
import torch

from typing import Union

# framework for using Llama
from llama_index import (
    VectorStoreIndex, 
    ServiceContext, 
    LangchainEmbedding, 
    SimpleDirectoryReader
)
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
# generate embeddings using HuggingFace
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import BitsAndBytesConfig

access_token = os.getenv("HF_KEY")
BNB_CONFIG = BitsAndBytesConfig(
    llm_int8_enable_fp32_cpu_offload=True,
    load_in_4bit = True, 
    bnb_4bit_quant_type = 'nf4', 
    bnb_4bit_use_double_quant = True, 
    bnb_4bit_compute_dtype = torch.bfloat16
)


class ChatApp:
    """
    Implementation of Llama2 as HuggingFaceLLM using llama_index
    """
    def __init__(self):
        # llama2 model to use
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"
        # embedding model
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    def define_llm_embeddings(self) -> Union[HuggingFaceLLM, LangchainEmbedding]:
        """
        function to define the HuggingFace LLM and the embeddings

        Return:
            llm (HuggingFaceLLM): HuggingFace LLM
            embeddings (LangchainEmbedding): model to generate embeddings
        """

        # define LLM model
        llm = HuggingFaceLLM(
            context_window = 4096,
            max_new_tokens = 256,
            generate_kwargs = {"temperature": 0.0, "do_sample": False},
            tokenizer_name = self.model_name,
            model_name = self.model_name,
            device_map = 'auto',
            # to run on CPU
            model_kwargs = {"torch_dtype": torch.float16 , "load_in_8bit":True, 'use_auth_token': access_token, 'quantization_config': BNB_CONFIG}
        )
        # get list of embeddings for document
        embeddings = LangchainEmbedding(
            HuggingFaceEmbeddings(
            model_name = self.embedding_model, 
            # to run model on CPU
            model_kwargs = {'device': 'cpu'}
            )
        )
        return llm, embeddings

    def load_document(self, file_name: str) -> VectorStoreIndex:
        """
        read the document and save to vector datastore

        Args:
            file_name (str): path to file

        Returns:
            index (VectorStoreIndex): vector datastore
        """
        
        # get file extension
        file_extension = file_name.split(".")[-1]
        if not file_extension in ["pdf", "docx", "csv"]:
            print(f"Cannot process {file_extension}. Only 'pdf', 'docx' and 'csv' are supported")
            return None
        
        # read the contents of directory
        directory = os.path.dirname(file_name)
        data = SimpleDirectoryReader(f"{directory}/").load_data()

        try:
            # get llm model and embedding model
            llm, embeddings = self.define_llm_embeddings()
            # create a service_context
            service_context = ServiceContext.from_defaults(
                chunk_size = 1024, 
                llm = llm, 
                embed_model = embeddings
            )
            # save document and its embeddings to vector store
            index = VectorStoreIndex.from_documents(
                documents = data, 
                service_context = service_context
            )
            return index
        except Exception as error:
            print(f"Exception while handling error :: Exception :: {str(error)}")
            return None

    def get_answer(self, fileobj: gradio.File, search_query: str) -> Union[str, str]:
        """
        get answers to user questions from Llama2 using API

        Args:
            fileobj (gradio.File) : uploaded file
            search_query (str): user question
        
        Returns:
            answer (str): answer from the LLM   
            sources (list): answer source from document
        """

        answer, sources = 'could not generate an answer', ''
        
        db = self.load_document(file_name=fileobj.name)
        if db:
            # perform chat
            query_engine = db.as_query_engine()
            try:
                llm_response = query_engine.query(search_query)
                # answer to query
                answer = llm_response.response
                # TODO- answer sources 
                sources = llm_response.get_formatted_sources()
            except Exception as error:
                print(f"Error while generating answer :: Exception :: {str(error)}")
        
        return answer, sources


def gradio_interface(inputs: list=[gradio.File(label = "Input file", file_types = [".pdf", ".csv", ".docx"]), 
                                   gradio.Textbox(label = "your input", lines = 3, placeholder = "Your search query ...")], 
                     outputs: list=[gradio.Textbox(label = "response", lines = 6, placeholder = "response returned from llama2 ...."),
                                    gradio.Textbox(label = "response source", lines = 6, placeholder = "source to response ...")]):
    """
    render a gradio interface

    Args:
        inputs (list): interface input components
        outputs (list): output components
    """

    chat_ob = ChatApp()
    demo = gradio.Interface(fn = chat_ob.get_answer, inputs = inputs, outputs = outputs)
    demo.launch(share = False)
    # uncomment for public URL (accessible for 3 days, deploy and its accessible for a lifetime)
    # demo.launch(share = True)


if __name__ == '__main__':

    gradio_interface()
