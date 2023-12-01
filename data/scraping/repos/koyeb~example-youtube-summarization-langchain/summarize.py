import os
import time
from langchain.chains import MapReduceDocumentsChain, LLMChain, ReduceDocumentsChain, StuffDocumentsChain
from langchain.document_loaders import (TextLoader)
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter


def summarize_transcript(filename):
    # Load transcript
    loader = TextLoader(filename)
    docs = loader.load()

    # Load LLM
    config = {'max_new_tokens': 4096, 'temperature': 0.7, 'context_length': 4096}
    llm = CTransformers(model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                        model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                        config=config,
                        threads=os.cpu_count())

    # Map template and chain
    map_template = """<s>[INST] The following is a part of a transcript:
    {docs}
    Based on this, please identify the main points. 
    Answer:  [/INST] </s>"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce template and chain
    reduce_template = """<s>[INST] The following is set of summaries from the transcript:
    {doc_summaries}
    Take these and distill it into a final, consolidated summary of the main points. 
    Construct it as a well organized summary of the main points and should be between 3 and 5 paragraphs.
    Answer:  [/INST] </s>"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )
    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )
    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=True,
    )

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)

    # Run the chain
    start_time = time.time()
    result = map_reduce_chain.__call__(split_docs, return_only_outputs=True)
    print(f"Time taken: {time.time() - start_time} seconds")
    return result['output_text']
