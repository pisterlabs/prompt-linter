from GuidelineService import GuidelineService
import html2text
from os import path
import tempfile
import openai

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from uuid import uuid4

import pinecone

htmlToText = html2text.HTML2Text()

openai_api_key = "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"
guideline_id = "s3-leitlinie-diagnostik-therapie-und-nachsorge-des-nierenzellkarzinoms"

service = GuidelineService()
guideline_data = service.download_guideline(guideline_id=guideline_id, guideline_state="published")

current_chapter_tree = ["", "", "", "", "", ""]
current_chapter_level = -1

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
pinecone.init(api_key="a5b5539f-8e83-4b74-ab9f-aa4a1da90fd9", environment="us-west1-gcp")
index = pinecone.Index("test-index")

def upsert_subsections(subsections):
    global current_chapter_level
    global htmlToText
    global text_splitter
    global index

    batch_limit = 100

    current_chapter_level += 1
    for subsection in subsections:
        if subsection['type'] == "ChapterCT":
            current_chapter_tree[current_chapter_level] = subsection['title']
        else:
            if 'text' in subsection:
                texts = []
                metadatas = []

                metadata = {'source': guideline_data['short_title'], 'chapter':",".join([x for x in current_chapter_tree if len(x)>0])}

                clean_text = htmlToText.handle(subsection['text']).strip()
                docs = text_splitter.split_text(clean_text)
                doc_metadatas = []
                for i in range(len(docs)):
                    item = {"chunk": i, "text": docs[i]}
                    item.update(metadata)
                    doc_metadatas.append(item)

                texts.extend(docs)
                metadatas.extend(doc_metadatas)

                ids = [str(uuid4()) for _ in range(len(texts))]
                embeds = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_documents(texts)
                index.upsert(vectors=zip(ids, embeds, metadatas))

        if 'subsections' in subsection and len(subsection['subsections']) > 0:
            upsert_subsections(subsection['subsections'])

    current_chapter_tree[current_chapter_level] = ""
    current_chapter_level -= 1

upsert_subsections(guideline_data['subsections'])
