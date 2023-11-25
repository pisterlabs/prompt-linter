from langchain.chains import RetrievalQA, VectorDBQA
from langchain.document_loaders import TextLoader
from InstructorEmbedding import INSTRUCTOR
import textwrap

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain

from lc.model import load_t5_model, build_t5
from lc.database import GetRetriever

from lc.database import get_docs, get_individual_docs

import os
import pandas as pd

class DefineChain():
    def __init__(self, data="transportation") -> None:
        self.data = data

    def llm_chain(self, query, k=3):
        model, tokenizer = load_t5_model()
        t5_pipe = build_t5(model, tokenizer)
        gr = GetRetriever(self.data)
        rt = gr.retrieve_data(k=k) 

        # Create the chain
        qna_chain = RetrievalQA.from_chain_type(llm=t5_pipe,
                                                chain_type="stuff",
                                                retriever=rt,
                                                return_source_documents=True)
        
        output = qna_chain(query)
        source = output["source_documents"]

        context = ""
        sources = []
        for s in source:
            context += s.page_content
            sources.append(s.metadata['source'])
        print("###########################################################")
        print("Question: " + str(query))
        print("###########################################################")
        print("The answer is derived from the following Context: " + context)
        print("###########################################################")
        print("The answer is derived from the following Context: " + str(sources))
        print("###########################################################")
        answer = output["result"]


        return answer
    
    def summarizer_chain(self):
        data = 'data_' + str(self.data)
        docs_dict = get_docs(data)

        # List of PDFs
        keys = list(docs_dict.keys())

        model, tokenizer = load_t5_model()
        t5_pipe = build_t5(model, tokenizer)

        # print(t5_pipe.get_num_tokens(doc[0].page_content))
        summarized_pdfs = {}
        for pdf in keys:
            doc = docs_dict[pdf]
            chain = load_summarize_chain(llm=t5_pipe, chain_type='map_reduce')
            summary = chain.run(doc)
            summarized_pdfs[pdf] = summary

        return summarized_pdfs


    # def individual_chain(self):
    #     data = 'data_' + str(self.data)

    #     docs_dict = get_individual_docs(data)

    #     # List of PDFs
    #     keys = list(docs_dict.keys())

    #     model, tokenizer = load_t5_model()
    #     t5_pipe = build_t5(model, tokenizer)

    #     # print(t5_pipe.get_num_tokens(doc[0].page_content))
    #     summarized_pdfs = {}
    #     for pdf in keys:
    #         doc = docs_dict[pdf]
    #         chain = load_summarize_chain(llm=t5_pipe, chain_type='map_reduce')
    #         summary = chain.run(doc)
    #         summarized_pdfs[pdf] = summary

    #     return summarized_pdfs

        
    def get_summary(self, query):
        model, tokenizer = load_t5_model()
        t5_pipe = build_t5(model, tokenizer)

        folder = "csvs"
        csv_files = os.listdir(folder)
        queries = []
        papers = []
        summaries = []

        for file in csv_files:
            df = pd.read_csv(os.path.join(folder, file))
            queries.append(df["query"].iloc[0])
            papers.append(df["paper"].iloc[0])
            summaries.append(df["summary"].iloc[0])

        N = len(csv_files)

        template = """Given N = {N} papers in list {papers}, along with their respective key findings in list {summaries} please answer the following question {query}"""
        prompt = PromptTemplate(template=template, input_variables=["N", "papers", "summaries", "query"])
        llm_chain = LLMChain(prompt=prompt, llm=t5_pipe)


        answer = llm_chain.predict(N=N, papers=papers, summaries=summaries, query=query)        

        return answer
