import os
import uuid
import datetime
import openai
from dotenv import load_dotenv
import numpy as np
import faiss
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import sounddevice as sd
import wavio

# Load environment variables from .env file
load_dotenv()

# Access environment variables
PPLX_API_KEY = os.environ.get("PPLX_API_KEY")
os.environ["PPLX_API_KEY"] = PPLX_API_KEY

model_name="llama-2-70b-chat"

sys.path.append("./components")
sys.path.append("./constants")
import speech_to_text
from pymongo import MongoClient

# MongoDB setup
MONGO_URI = os.environ.get('MONGO_DB_URI')  # Your MongoDB connection string
client = MongoClient(MONGO_URI)
MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME')
MONGO_DB_COLLECTION = os.environ.get('MONGO_DB_COLLECTION')

db = client[MONGO_DB_NAME]
collection = db[MONGO_DB_COLLECTION]

# Fetch phrases and their corresponding audio codes
phrases_dict = {}
for document in collection.find({}):
    phrases_dict[document['phrase']] = document['audio_code']


def generate_unique_id():
    return str(uuid.uuid4())

conversation_id = generate_unique_id()

# 1. FAISS Indexing
# Convert phrases to embeddings for searching
vectorizer = TfidfVectorizer().fit(phrases_dict.keys())
phrases_embeddings = vectorizer.transform(phrases_dict.keys())
index = faiss.IndexIDMap(faiss.IndexFlatIP(phrases_embeddings.shape[1]))
index.add_with_ids(np.array(phrases_embeddings.toarray(), dtype=np.float32), np.array(list(range(len(phrases_dict)))))

def find_matching_audio(sentence):
    # Start the timer
    start_time_f = datetime.datetime.now()

    # Convert the sentence to embedding and search in the index
    sentence_embedding = vectorizer.transform([sentence]).toarray().astype(np.float32)
    D, I = index.search(sentence_embedding, 1)
    match_index = I[0][0]
    matching_sentence = list(phrases_dict.keys())[match_index]
    
    # Stop the timer
    end_time_f = datetime.datetime.now()
    elapsed_time_f = (end_time_f - start_time_f).total_seconds()

    # Print the elapsed time
    print(f"Time taken by Faiss to fetch similar answer: {elapsed_time_f:.6f} seconds")
    
    if D[0][0] > 0.1:  # You can adjust this threshold based on desired accuracy
        return phrases_dict[matching_sentence]
    return None

# 2. Sentence Splitting and 3. Play Audio
def handle_gpt_response(full_content):
    sentences = [s.strip() for s in full_content.split('.') if s]
    for sentence in sentences:
        audio_code = find_matching_audio(sentence)
        if audio_code:
            audio_path = f"./assets/audio_files_pixel/{audio_code}.wav"
            wav_obj = wavio.read(audio_path)
            sd.play(wav_obj.data, samplerate=wav_obj.rate)
            sd.wait()  # Wait until audio playback is done

def chat_with_user():
    messages = [
        {
            "role": "system",
            "content": (
                "Agent: Jacob. Company: Gadget Hub. Task: Demo. Product: Google Pixel. Features: Night Sight, Portrait Mode, Astrophotography, Super Res Zoom, video stabilization. Battery: All-day. Objective: Google Pixel over iPhone. Discuss features. Interest? Shop visit. Agree? Name, contact. Address inquiries. Details given? End: great day.you are Jacob. Only respond to the last query in short."
            ),
        }
    ]

    while True:
        query = speech_to_text.transcribe_stream()

        if query.lower() == "exit":
            break

        messages.append({"role": "user", "content": query})
        print(f"{{ {messages} }}")  # Print the chat history

        # Record the time when the question is given
        start_time = datetime.datetime.now()

        # Chat completion with streaming
        response_stream = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            api_base="https://api.perplexity.ai",
            api_key=PPLX_API_KEY,
            stream=True,
        )

        full_content = ""
        first_response = True
        for response in response_stream:
            if 'choices' in response:
                content = response['choices'][0]['message']['content']
                full_content = content  # Reset full_content to current content, instead of appending
                if first_response:
                    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                    print(f"Time taken for answer to start streaming: {elapsed_time:.2f} seconds")
                    first_response = False

        print(full_content)  # Print the final complete response
        handle_gpt_response(full_content)  # Handle GPT's response

        messages.append({"role": "assistant", "content": full_content})

if __name__ == "__main__":
    chat_with_user()
