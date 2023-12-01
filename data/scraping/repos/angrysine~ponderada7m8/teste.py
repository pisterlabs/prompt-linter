from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import  RunnablePassthrough
from langchain.document_loaders import TextLoader, UnstructuredURLLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import gradio as gr


# load the document and split it into chunks
loader = TextLoader("./teste.txt")

# load the document and split it into chunks

documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1169, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
vectorstore = Chroma.from_documents(docs, embedding_function)

retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = Ollama(model="ponderada")


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
)
def chat(text):
    output_text = ""
    for s in chain.stream(text):
        output_text+=s
            
        if "<|im_end|>" in output_text:
            output_text = output_text.removesuffix("<|im_end|>")
            break
        
    return output_text
# while True:
#     text = input("ask your question Nicola: ")
#     if text == "exit":
#         break
#     print(chat(text))

demo = gr.Interface(fn=chat, inputs="text", outputs="text")
    
if __name__ == "__main__":
    demo.launch(show_api=False)   