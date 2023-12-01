import os
from flask import Flask, request, jsonify
from translate import Translator as MicrosoftTranslator
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, DirectoryLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = ""

translator = MicrosoftTranslator(from_lang='en', to_lang='en')

# Load and process the text files
loader = DirectoryLoader('dataset', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()

# Splitting the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Embed and store the texts
persist_directory = 'db'
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
vectordb.persist()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever = vectordb.as_retriever()
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

def english_to_hinglish(english_sentence):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Translate the following English sentence to Hinglish: '{english_sentence}'",
        max_tokens=100,
        temperature=0.0,
        n=1,
        stop=None
    )

    hinglish_translation = response.choices[0].text.strip()
    return hinglish_translation

def translate_to_target_language(text, target_language):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Translate the following text to {target_language}: '{text}'",
        max_tokens=100,
        temperature=0.0,
        n=1,
        stop=None
    )

    translated_text = response.choices[0].text.strip()
    return translated_text

@app.route('/answer', methods=['POST'])
def ask_question():
    if request.method == 'POST':
        data = request.get_json()
        question = data.get('question')
        target_language = data.get('target_language', 'ar')  # Default to Arabic if target language is not specified

        engineer_prompt = "provide me an answer in around 20-25 words"
        query = engineer_prompt + " " + question

        llm_response = qa_chain(query)
        answer = llm_response['result']

        translated_answer = translate_to_target_language(answer, target_language)
        translated_answer_hinglish = english_to_hinglish(answer)

        response = {
            "answer": answer,
            "translated_answer": translated_answer
            # "translated_answer_hinglish": translated_answer_hinglish
        }

        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=6971)