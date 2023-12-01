from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import VectorDBQA
from langchain.document_loaders import DirectoryLoader
import openai
import random
import pyttsx3
import os

openai.api_key = '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'

os.environ["OPENAI_API_KEY"] = "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"
engine = pyttsx3.init()

llm = openai.Completion.create(
    model="text-davinci-003",
    # prompt="write a mail to your boss for resignation\n\nSubject: Resignation\n\nDear [Name], \n\nI write this letter to officially notify you of my resignation from my position at [Company Name].\n\nThis was a difficult decision to make, but I feel the need to pursue other professional opportunities that will help me further develop my skills.\n\nI'm grateful for the time I have had here and the experience I have gained. I would like to thank you for the opportunities you have given me, especially [x], which have enabled me to become the person I am today.\n\nAs per our employment agreement, I will be available to assist in the transition of my role until [date], at which time I will have completed my full notice period. \n\nOnce again, thank you for everything and I wish you all the best for the future of the company.\n\nSincerely, \n\n[Your Name]",
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

loader = DirectoryLoader('C:\\Users\\kashi\\Downloads\\chat', glob="**/*.txt")
documents = loader.load()
result2 = documents[0].page_content
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
max_text_length = 1000  # Maximum length of each segment

# Split longer texts into smaller segments
segmented_texts = []
for document in documents:
    content = document.text
    if len(content) > max_text_length:
        num_segments = len(content) // max_text_length
        segments = [content[i:i + max_text_length] for i in range(0, len(content), max_text_length)]
        segmented_texts.extend(segments)
    else:
        segmented_texts.append(content)

texts = text_splitter.split_documents(segmented_texts)
embeddings = OpenAIEmbeddings(chunk_size=1)
docsearch = Chroma.from_documents(texts, embeddings)

llm2 = openai.Completion.create(model="text-davinci-003")

qa = VectorDBQA.from_chain_type(llm=llm2, chain_type="stuff", vectorstore=docsearch)

result = qa.run("Ask your question here")
print(result)
result.replace("/n", "<br/>")
