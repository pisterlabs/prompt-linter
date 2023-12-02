
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import config
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import  FAISS



def mcb(prompt):
    reader = PdfReader('./content/brochure.pdf')
    os.environ["OPENAI_API_KEY"] = config.open_ai_key
        # read data from the file and put them into a variable called raw_text
    raw_text = ''
    print("i am in ",prompt)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits.

    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 2000,
        chunk_overlap  = 100,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()

    docsearch = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    # query = "tell me about PERIOFLOW APPLICATIONS"
    query=prompt
    query1=prompt+" answer should contain only 20 word"
    docs = docsearch.similarity_search(query)
    ans=chain.run(input_documents=docs, question=query1)
    print(ans," answer from open ai")
    return ans










