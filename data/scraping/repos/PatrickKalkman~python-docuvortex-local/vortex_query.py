from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores.chroma import Chroma

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline

from settings import COLLECTION_NAME, PERSIST_DIRECTORY


class VortexQuery:
    """Handles querying using the Vortex mechanism."""

    def __init__(self):
        """Initialize the VortexQuery class."""
        self.chain = self._initialize_chain()

    def _initialize_chain(self):
        """Initialize the retrieval chain.

        Returns:
            RetrievalQA: A configured retrieval chain.
        """
        tokenizer = LlamaTokenizer.from_pretrained("TheBloke/wizardLM-7B-HF")
        model = LlamaForCausalLM.from_pretrained("TheBloke/wizardLM-7B-HF",
                                                 load_in_8bit=True,
                                                 device_map='auto',
                                                 torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True
                                                 )

        generation_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=1024,
            temperature=0,
            top_p=0.95,
            repetition_penalty=1.15
        )

        local_llm = HuggingFacePipeline(pipeline=generation_pipe)

        instructor_embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl",
            model_kwargs={"device": "cuda"}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=instructor_embeddings,
            persist_directory=PERSIST_DIRECTORY,
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        return RetrievalQA.from_chain_type(llm=local_llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True)

    def ask_question(self, question: str):
        """Ask a question through the retrieval chain.

        Args:
            question (str): The question to be asked.

        Returns:
            Response: The response from the retrieval chain.
        """
        return self.chain(question)
