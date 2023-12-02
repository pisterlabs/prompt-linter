import tiktoken
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

prompt_template = """Use the following pieces of context to answer the question at the end.

{context}

Question: {question}
Helpful Answer:"""  


def compute_tokens_length(s):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(s))


class Wikipedia:
    def __init__(self, model_name="gpt-3.5-turbo", embedding_model_name="text-embedding-ada-002", chunk_size=384, chunk_overlap=64, lang='en'):
        self.client = ChatOpenAI(model_name=model_name)
        self.embeddings = OpenAIEmbeddings(model=embedding_model_name)
        self.loader = WikipediaLoader
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.lang = lang
        self.template = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
    def run(self, input: str, question: str):
        """_summary_

        Args:
            input (str): input

        """        
        try:
            loader = self.loader(query=input, load_max_docs=6, lang=self.lang)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap,
                length_function=compute_tokens_length,
            )
            all_splits = text_splitter.split_documents(data)
            vectorstore = Chroma.from_documents(documents=all_splits, embedding=self.embeddings)
            qa_chain = RetrievalQA.from_chain_type(
                self.client,
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": self.template}
            )
            result = qa_chain({"query": question})
                
            return result["result"]
        except Exception as e:
            return e
