import os, json 

from langchain.prompts import PromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.document_loaders import * 
from langchain.docstore.document import Document

from prompts import * 

class DocumentReader(object):
    """ This class loads a document. """
    def __init__(self, db_dir='db', chunk_size=1000, chunk_overlap=0):
        self.db_dir = db_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.summary_dir = os.path.join(self.db_dir, "summaries")
        os.makedirs(self.summary_dir, exist_ok=True)
        
    def load(
            self, doc_path, text=None,
            collection_name='langchain',
            db_dir='db', chunk_size=1000, chunk_overlap=0,
            debug=False,
            ):
        """ This function loads a document. """
        # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if text is not None and len(text) > 0:
            # convert the text into a Document object
            # text_doc = Document(page_content=text)
      
            texts = text_splitter.split_text(text)
            chunks = [Document(page_content=chunk) for chunk in texts]
        else:
            loader = UnstructuredFileLoader(doc_path)
            # if os.path.splitext(doc_path)[1] in ['.doc', '.docx']:
            #     loader = UnstructuredWordDocumentLoader(doc_path)
            # elif os.path.splitext(doc_path)[1] in ['.pdf']:
            #     loader = UnstructuredPDFLoader(doc_path)
            # else:
            #     raise ValueError("Invalid document type: {}".format(os.path.splitext(doc_path)[1]))
            documents = loader.load()
            
            chunks = text_splitter.split_documents(documents)

        embedding = OpenAIEmbeddings(model='text-embedding-ada-002')

        # create the vector store (if it doesn't exist)
        if debug:
            vectordb = Chroma(persist_directory=db_dir, embedding_function=embedding, collection_name=collection_name)
        else:
            vectordb  = Chroma.from_documents(
                documents=chunks, embedding=embedding, persist_directory=db_dir, collection_name=collection_name
                )

        return chunks, vectordb
    
    def summarize(
            self, chunks, 
            # map_prompt_template=MAP_PROMPT_TEMPLATE,
            # combine_prompt_template=COMBINE_PROMPT_TEMPLATE,
            templates,
            summary_option="map_reduce",
            temperature=0.0, max_tokens=1000,
            debug=False,
            ):
        """ This function summarizes a document. """
        # save the summaries
        if summary_option == "map_reduce":
            save_path = os.path.join(self.summary_dir, "total_summary.json")
        elif summary_option == "refine":
            save_path = os.path.join(self.summary_dir, "total_refine.json")
        elif summary_option == "translate":
            save_path = os.path.join(self.summary_dir, "total_translate.json")
        else:
            raise ValueError("Invalid summary option: {}".format(summary_option))
        
        if debug and os.path.exists(save_path):
            with open(save_path, "r") as f:
                data = json.load(f)
            return data["total_summary"], data["chunk_summaries"]
        
        # Setup the LLM
        llm = ChatOpenAI(
            model_name='gpt-3.5-turbo',
            temperature=temperature,
            max_tokens=max_tokens,
            )
        
        if summary_option == "map_reduce":
            map_prompt_template = templates['map_prompt_template']
            map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

            combine_prompt_template = templates['combine_prompt_template']
            combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

            chain = load_summarize_chain(
                llm=llm, chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt,
                verbose=True, 
                return_intermediate_steps=True,
            )
        elif summary_option == "refine":
            initial_prompt_template = templates['refine_initial_prompt_template']
            initial_prompt = PromptTemplate(template=initial_prompt_template, input_variables=["text"])

            refine_prompt_template = templates['refine_prompt_template']
            refine_prompt = PromptTemplate(template=refine_prompt_template, input_variables=["existing_answer", "text"])

            chain = load_summarize_chain(
                llm=llm, chain_type="refine",
                question_prompt=initial_prompt,
                refine_prompt=refine_prompt,
                verbose=True, 
                return_intermediate_steps=True,
            )
        elif summary_option == "translate":
            translate_prompt_template = templates['translate_prompt_template']
            translate_prompt = PromptTemplate(template=translate_prompt_template, input_variables=["text"])

            combine_prompt_template = templates['combine_prompt_template']
            combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

            chain = load_summarize_chain(
                llm=llm, chain_type="map_reduce",
                map_prompt=translate_prompt,
                combine_prompt=combine_prompt,
                verbose=True, 
                return_intermediate_steps=True,
            )
        else:
            raise ValueError("Invalid summary option: {}".format(summary_option))

        result = chain({"input_documents": chunks})
        
        total_summary = result["output_text"]

        chunk_summaries = []
        for chunk_doc, chunk_summary in zip(chunks, result["intermediate_steps"]):
            chunk_summaries.append({'chunk_content': chunk_doc.page_content, 'chunk_summary': chunk_summary})

        with open(save_path, "w") as f:
            data = {
                "total_summary": total_summary,
                "chunk_summaries": chunk_summaries,
            }
            json.dump(data, f, indent=4, ensure_ascii=False)

        return total_summary, chunk_summaries
    
    def ask(
            self, query, vectordb, templates,
            temperature=0.0, max_tokens=1000,
            debug=False,
            ):
        """ This function asks a question and returns the answer from the document. """
        llm = ChatOpenAI(
            model_name='gpt-3.5-turbo',
            temperature=temperature,
            max_tokens=max_tokens,
            )
        
        query_prompt_template = templates['query_prompt_template']
        query_prompt = PromptTemplate(
            template=query_prompt_template, input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", 
            retriever=vectordb.as_retriever(),
            # memory=memory,
            chain_type_kwargs={"prompt": query_prompt},
            verbose=True,
            return_source_documents=True,
            )

        result = chain({"query": query})

        answer = result["result"]
        source_chunks = result["source_documents"]

        return answer, source_chunks

    def translate(
            self, chunks,
            templates,
            temperature=0.0, max_tokens=None,
            debug=False
            ):  
        """ This function translates a document. """

        # Setup the LLM
        llm = ChatOpenAI(
            model_name='gpt-3.5-turbo',
            temperature=temperature,
            max_tokens=max_tokens,
            )
        
