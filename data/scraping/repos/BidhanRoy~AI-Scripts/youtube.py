import re, os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

def get_video_id(url):
	pattern = r'(?:https?:\/\/)?(?:www\.)?youtu(?:\.be\/|be\.com\/(?:watch\?v=|embed\/|v\/))([a-zA-Z0-9_-]{11})'
	match = re.match(pattern, url)
	return match.group(1) if match else None

def get_transcript(video_id):
	if not video_id:
		raise ValueError("Invalid YouTube URL")
	transcripts = YouTubeTranscriptApi.get_transcript(video_id)

	return [transcript['text'] for transcript in transcripts]

def get_vector_db(video_id, transcript):
    CHROMA_DB_PATH = "./chroma/{video_id}".format(video_id=video_id)

    if not os.path.exists(CHROMA_DB_PATH):
        # Create a new Chroma DB
        print(f'Creating Chroma DB at {CHROMA_DB_PATH}...')
        vector_db = Chroma.from_texts(transcript, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH)
        vector_db.persist()
    else:
        # Load an existing Chroma DB
        print(f'Loading Chroma DB from {CHROMA_DB_PATH}...')
        vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings())
    
    return vector_db

if __name__ == "__main__":
    print('\n\n\033[31m' + 'Input a youtube video link: ' + '\033[m')
    youtube_url = input()
    youtube_id = get_video_id(youtube_url)
    transcript = get_transcript(youtube_id)
    vector_db = get_vector_db(youtube_id, transcript)

    # Load a QA chain
    qa_chain = load_qa_chain(OpenAI(temperature=1), chain_type="stuff")
    
    # Create a RetrievalQA object using the QA chain and the retriever from vector_db
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=vector_db.as_retriever())

    while True:
        print('\n\n\033[31m' + 'Ask a question about your video' + '\033[m')
        user_input = input()
        print('\033[31m' + qa.run(user_input) + '\033[m')
