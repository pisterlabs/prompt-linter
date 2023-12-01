from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

import yt_dlp

def download_mp4_from_youtube(urls, job_id):
    video_info = []

    for i, url in enumerate(urls):
        file_name = f"{job_id}_{i}.mp4"

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': file_name,
            'quiet': False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            title = result.get("title", "")
            author = result.get("uploader", "")

        video_info.append((file_name, title, author))


    return video_info

urls=["https://www.youtube.com/watch?v=mBjPyte2ZZo&t=78s",
    "https://www.youtube.com/watch?v=cjs7QKJNVYM",]
video_details = download_mp4_from_youtube(urls, 1)


import whisper

model = whisper.load_model("base")

results = []

for video in video_details:
    result = model.transcribe(video[0], verbose=True)
    results.append(result["text"])
    print(f"Transcription for {video[1]} by {video[2]}: {result['text']}\n")

with open("transcriptions.txt", "w") as f:
    f.write("\n".join(results))

from langchain.text_splitter import RecursiveCharacterTextSplitter

with open("transcriptions.txt", "r") as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    separators=[" ", ",", "\n"]
)
texts = text_splitter.split_text(text)

from langchain.docstore.document import Document

docs = [Document(page_content=text) for text in texts]

from langchain.vectorstores import DeepLake
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = OpenAI(model_name="text-davinci-003", temperature=0.0)

activeloop_dataset_name = "langchain_course_6_yt"
dataset_path = f"hub://{os.environ.get('ACTIVELOOP_ORGID')}/{activeloop_dataset_name}"
vecdb = DeepLake(dataset_path=dataset_path, embedding=embeddings)

vecdb.add_documents(docs)

retriever = vecdb.as_retriever(
    distance_metric="cos",
    k=4,
)

from langchain.prompts import PromptTemplate

PROMPT = PromptTemplate(
    template="""Use the following pieces of transcripts from a video to answer the question in bullet points and summarized. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Summarized answer in bullter points:""",
    input_variables=["context", "question"],
)

from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
)

print(qa.run("Summarize the mentions of google according to their AI program"))