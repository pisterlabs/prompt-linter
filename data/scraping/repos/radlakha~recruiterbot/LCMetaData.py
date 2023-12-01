from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PagedPDFSplitter
import os

from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory

# Dont need the below line if you have the key in your environment variables
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = ""   # OpenAI Key

#Import data
def get_pdf_data(doc_name): #Function which loads PDF files (Along with page numbers). Name is required rn, can be made optional.
    f_doc_name = 'C:/Users/tsohani/Project/ChatBot/Resumes/' + doc_name
    loader = PagedPDFSplitter(f_doc_name)
    pages = loader.load_and_split()

    return(pages)

sources = [                      #List of all the sources to be used
    get_pdf_data("software.pdf"),
    get_pdf_data("Kirthi_Nagane.pdf"),
    get_pdf_data("Vidhyasimhan_J_Resume.pdf"),
    get_pdf_data("Sujeet Neema - Cognitive Developer.pdf"),
    
]

#Create chunks
source_chunks = []  #Creating chunks of the source documents    
splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
for source1 in sources: #For each of the source mentioned
    for source in source1: #Go to each page
        for chunk in splitter.split_text(source.page_content): #Chunk up the content in each page
            chunk = chunk + "Filename: " + source.metadata['source']
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))#Append it to source_chunks

#Create embeddings
search_index = FAISS.from_documents(source_chunks, HuggingFaceEmbeddings())   #Simiarity Search with HuggingFaceEmbeddings

template = """You are a recruiter chatbot having a conversation with a human.

Given the following extracted parts of a long document and a question, create a final answer.
Use only the information available in the context. Don't make up an answer.
Do not modify contact details. Give the filename too. 

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], 
    template=template
)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

#Create LLM Chain
chain2 = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", memory = memory, prompt = prompt)

 
def print_answer2(question):
    
    return (chain2(
            {
                "input_documents": search_index.similarity_search(question, k=4),
                "human_input": question,                
            },
            return_only_outputs=True,
            )['output_text']
            )

# question ="Who would be suitable for the position of a voip engineer."
# while(question !="0" and len(question)>5):
#     print_answer2(question)
#     question = input("next question?")

