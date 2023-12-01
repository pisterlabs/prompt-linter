# PDF Knowledge Base using vectors


# pip install packages
#!pip install -q langchain==0.0.150 pypdf pandas matplotlib tiktoken textract transformers openai faiss-cpu streamlit


# import packages
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain


# Set api key   
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"


# You MUST add your PDF to local files in order to use this method
# Simple method - Split by pages 
loader = PyPDFLoader("./SMILES-Related.pdf")
pages = loader.load_and_split()
# print(pages[0])

# Split by paragraphs
# chunks = pages


# Advanced method - Split by chunk
# Step 1: Convert PDF to text
import textract
doc = textract.process("./SMILES-Related.pdf")

# Step 2: Save to .txt and reopen (helps prevent issues)
with open('SMILES-Related-2.txt', 'w') as f:
    f.write(doc.decode('utf-8'))

with open('SMILES-Related-2.txt', 'r') as f:
    text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

# Quick data visualization to ensure chunking was successful
# Create a list of token counts
token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

# Create a DataFrame from the token counts
df = pd.DataFrame({'Token Count': token_counts})

# Create a histogram of the token count distribution
df.hist(bins=40, )

# Show the plot
plt.show()



# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)


# Create conversation chain that uses our vectordb as retriver, this also allows for chat history management
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())


chat_history = []

def on_submit(_):
    query = input_box.value
    input_box.value = ""

    if query.lower() == 'exit':
        print("Thank you for using Neal's chatbot!")
        return

    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))

    display(widgets.HTML(f'<b>User:</b> {query}'))
    display(widgets.HTML(f'<b><font color="blue">Chatbot:</font></b> {result["answer"]}'))

print("Ask me a question about the PDF document you just uploaded. Type 'exit' to stop.")

input_box = widgets.Text(placeholder='Please enter your question here:')
input_box.on_submit(on_submit)

display(input_box)