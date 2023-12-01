import nltk
import requests

nltk.download("stopwords")
import numpy as np
import io
import openai
import pdfplumber
from flask import Flask, request, jsonify, send_from_directory
from openai.embeddings_utils import get_embedding
from cachetools import cached, LRUCache
import hashlib
import pytesseract
import fitz
import pinecone
from PIL import Image
from io import BytesIO
from langdetect import detect
import re
import spacy
import time
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from openai.embeddings_utils import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from transformers import GPT2Tokenizer
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
#import en_core_web_sm
nlp = spacy.load("en_core_web_sm")
llm = ChatOpenAI(model_name='gpt-4', temperature=0.4, max_tokens=4000, request_timeout=120)

pdf_cache = LRUCache(maxsize=100)  # Adjust the maxsize according to your requirements
nltk.download('punkt')
app = Flask(__name__)


def test_openai_connection():
    try:
        openai.Engine.list()
        print("Connected to OpenAI successfully.")
    except Exception as e:
        print("Failed to connect to OpenAI:", e)
        exit(1)


def generate_cache_key(pdf_content):
    pdf_hash = hashlib.md5(pdf_content).hexdigest()
    return pdf_hash


# Add a function to detect the language of the text
def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print(f"Error detecting language: {e}")
        return None


def translate_text(text, src_lang, dest_lang):
    if src_lang == dest_lang:
        return text

    prompt = f"Translate the following text from {src_lang} to {dest_lang}: '{text}'"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.3,
    )

    translated_text = response.choices[0].text.strip()
    if not translated_text:
        return text  # Return the original text if translation is empty or None
    return translated_text


from flask_cors import CORS

# Add CORS support for your Flask app
CORS(app)


@app.route('/count_tokens', methods=['POST'])
def count_tokens_endpoint():
    text = request.form['text']
    model = request.form['model']

    token_count = len(word_tokenize(text))

    return jsonify(token_count=token_count)


def extract_images_and_text_from_pdf(pdf_content):
    images = []
    texts = []

    # Load the PDF with PyMuPDF
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    # Iterate through the pages
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Extract the images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Open the image with PIL and perform OCR
            pil_image = Image.open(io.BytesIO(image_bytes))
            image_text = pytesseract.image_to_string(pil_image)
            if image_text.strip():
                images.append(pil_image)
                texts.append(image_text)

    return images, texts


@cached(pdf_cache, key=generate_cache_key)
def tokenize_pdf(pdf_content):
    with pdfplumber.open(BytesIO(pdf_content)) as pdf:
        num_pages = len(pdf.pages)
        text = "\n".join(page.extract_text() for page in pdf.pages)
        if not text:
            raise ValueError("Text extraction failed. The PDF may not have selectable text.")

        # Extract images and OCR text
        images, ocr_texts = extract_images_and_text_from_pdf(BytesIO(pdf_content))

        # Combine the selectable text and OCR text
        combined_text = text + "\n" + "\n".join(ocr_texts)

    tokens = nltk.word_tokenize(combined_text)
    return combined_text, tokens, num_pages


@app.route('/extract_text', methods=['POST'])
def extract_text():
    pdf_content = request.files['pdf'].read()

    try:
        text, tokens, num_pages = tokenize_pdf(pdf_content)
    except Exception as e:
        print("Text extraction failed:", e)
        return jsonify({"error": "Text extraction failed. Please try again with a different PDF file."}), 400

    print("Text extraction completed.")
    print("Extracted text:", text[:100], "...")  # Print only the first 100 characters to avoid cluttering logs
    return jsonify(text=text, num_pages=num_pages)


max_response_tokens = 100


# max_chunk_tokens = 3072 - max_response_tokens  # 3072 is used instead of 4096 to ensure a larger buffer


# Add a new function to calculate temperature and max_tokens
def calculate_params(num_pages, question_length):
    # Adjust temperature based on the number of pages
    if num_pages < 50:
        temperature = 0.3
    else:
        temperature = 0.4

    # Adjust max_tokens based on the question length
    if question_length < 50:
        max_tokens = 4200 - 400  # Leaving some tokens for the system message
    elif question_length < 100:
        max_tokens = 4200 - 550
    else:
        max_tokens = 4200 - 600

    return temperature, max_tokens


from nltk.tokenize import word_tokenize


def count_tokens(text):
    token_count = 0
    try:
        tokens = list(word_tokenize(text))
        token_count = len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")

    return token_count


def truncate_text(text, max_tokens, question_tokens, system_tokens, reserve_response_tokens):
    available_tokens = max_tokens - question_tokens - system_tokens - reserve_response_tokens

    tokens = word_tokenize(text)
    truncated_tokens = []
    current_token_count = 0

    for token in tokens:
        current_token_count += 1

        if current_token_count > available_tokens:
            break

        truncated_tokens.append(token)

    return ' '.join(truncated_tokens)


def truncate_summary_text(text, available_tokens):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.tokenize(text)
    truncated_tokens = tokens[:available_tokens]
    return tokenizer.decode(truncated_tokens)


def sliding_window_chunk(text, window_size):
    chunks = []
    text_length = len(text)
    start = 0

    while start < text_length:
        end = start + window_size
        if end >= text_length:
            end = text_length
        else:
            last_space = text.rfind(' ', start, end)
            if last_space != -1:
                end = last_space

        chunk = text[start:end].strip()
        chunks.append(chunk)
        start = end

    return chunks


def extract_entities(text, num_keywords=5):
    # Remove all non-alphanumeric characters and replace them with spaces
    text = re.sub(r'[^\w\s]', ' ', text)

    # Process the text using spaCy
    doc = nlp(text)

    # Extract noun chunks and lemmatize them
    lemmatized_chunks = [
        chunk.lemma_ for chunk in doc.noun_chunks
        if chunk.root.pos_ not in {"PRON", "DET"}
    ]

    # Count the frequency of each lemmatized chunk
    chunk_counter = Counter(lemmatized_chunks)

    # Extract the most common keywords
    common_keywords = chunk_counter.most_common(num_keywords)

    # Return the list of keywords
    return [keyword for keyword, count in common_keywords]


def preprocess_text(text):
    # Remove all non-alphanumeric characters and replace them with spaces
    text = re.sub(r'[^\w\s]', ' ', text)

    # Convert the text to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stopwords]

    # Join the tokens back into a single string
    text = ' '.join(tokens)

    return text


# Initialize the keyword cache
keyword_cache = {}


def clear_keyword_cache():
    keyword_cache.clear()


# Function to get keywords from cache or extract them if not found
def get_keywords(text, num_keywords=5):
    # Pre-process the text
    processed_text = preprocess_text(text)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Calculate the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform([processed_text])

    # Get the feature names
    feature_names = vectorizer.get_feature_names_out()

    # Get the TF-IDF scores for each feature
    tfidf_scores = tfidf_matrix.toarray().flatten()

    # Sort the features by their TF-IDF scores
    sorted_indices = tfidf_scores.argsort()[::-1]

    # Extract the top num_keywords features
    top_keywords = [feature_names[i] for i in sorted_indices[:num_keywords]]

    return top_keywords


def extract_entities_large_text(text, max_chunk_tokens=4090):
    # Split the text into smaller chunks
    text_chunks = sliding_window_chunk(text, max_chunk_tokens)
    all_entities = []

    for chunk in text_chunks:
        # Extract entities for each chunk
        chunk_entities = get_keywords(chunk)
        print(chunk_entities)
        all_entities.extend(chunk_entities)

    # Combine the extracted entities and remove duplicates
    unique_entities = list(set(all_entities))
    return unique_entities


def get_embeddings(texts):
    embeddings = []

    for text in texts:
        embedding = get_embedding(text, engine='text-embedding-ada-002')
        embeddings.append(embedding)

    return embeddings


def calculate_file_hash(file):
    file.seek(0)
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)
    return file_hash


def filter_relevant_chunks_v3(text_chunks, important_entities, question_embedding, text_chunk_embeddings, top_n=10):
    # Calculate the cosine similarities
    cosine_similarities = cosine_similarity(text_chunk_embeddings, question_embedding)
    print('cosine similarities are')
    print(cosine_similarities)

    # Get the indices of the cosine similarities sorted in descending order
    sorted_indices = np.argsort(cosine_similarities)[::-1]
    print(sorted_indices)

    # Select the top_n indices
    top_indices = sorted_indices[:top_n]

    print('top indices are')
    print(top_indices)

    # Get the top_n chunks based on the selected indices
    relevant_chunks = [text_chunks[i] for i in top_indices]
    print(relevant_chunks)

    return relevant_chunks


import difflib


def remove_duplicate_sentences(text_input, similarity_threshold=0.8):
    if isinstance(text_input, str):
        text_chunks = [text_input]
    elif isinstance(text_input, list):
        text_chunks = text_input
    else:
        raise ValueError("text_input must be a string or a list of strings")

    unique_sentences = []

    for text in text_chunks:
        sentences = nltk.sent_tokenize(text)

        for sentence in sentences:
            is_duplicate = False

            for unique_sentence in unique_sentences:
                similarity = difflib.SequenceMatcher(None, sentence, unique_sentence).ratio()

                if similarity > similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_sentences.append(sentence)

    return " ".join(unique_sentences)


def is_irrelevant_answer(answer, irrelevant_phrases):
    for phrase in irrelevant_phrases:
        if phrase.lower() in answer.lower():
            return True
    return False


def process_chunks(text_chunks, question, temperature, max_tokens, reserve_response_tokens, system_message,
                   summarized_chat_history):
    answers = []
    irrelevant_phrases = [
        "no information", "could not find any information", "please provide more information",
        "I'm sorry, but", "I apologize", "The provided text does not directly answer the question",
        "No information available"
    ]

    for chunk in text_chunks:
        # The hardcoded system_message has been removed

        system_tokens = count_tokens(system_message)

        message_tokens = count_tokens(question) + count_tokens(chunk) + system_tokens

        if message_tokens + reserve_response_tokens >= max_tokens:
            print("Token count exceeds the model's limit. Adjust the text or question.")
            return []

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": system_message + "and here is chat history for your knowledge but give more emphasis to text chunk than the history:" + summarized_chat_history
                },
                {"role": "user", "content": f"Based on the provided text, {question}"},
                {"role": "assistant", "content": f"Text chunk: {chunk}"}
            ],
            max_tokens=max_tokens - message_tokens - 1,  # Subtracting 1 to account for potential rounding errors
            n=1,
            stop=None,
            temperature=temperature,
            timeout=120
        )

        answer = response.choices[0].message['content'].strip()
        unique_answer = remove_duplicate_sentences(answer)

        if not is_irrelevant_answer(unique_answer, irrelevant_phrases):
            print("Answer for chunk:", unique_answer)
            answers.append(answer)
        else:
            answers.append("No information available")

    return answers


# def summarize_answers(answers, temperature, max_tokens, system_message, question):
#     # Remove similar answers
#     # unique_answers = list(set(answers))
#
#     # Join unique answers
#     # text = " ".join(unique_answers)
#     text = answers
#     print(text)
#
#     # Check if the question asks for a summary or key points
#     summary_keywords = ["summary", "key points", "key takeaways", "overview", "main points"]
#     summarized = any(re.search(r'\b{}\b'.format(kw), question, re.IGNORECASE) for kw in summary_keywords)
#
#     # Choose the appropriate system message based on the question type
#     if summarized:
#         prompt_system_message =  "here is the user question:" + question + "Summarise the text below as an answer to user's question:" """ \
#
#                          {text}"""
#     else:
#         prompt_system_message =  "here is the user question:" + question + "Combine the chunks below in a coherent manner as an answer to user's question:" """ \
#
#                          {text}"""
#
#     PROMPT = PromptTemplate(template=prompt_system_message, input_variables=["text"])
#     chain = LLMChain(llm=llm, prompt=PROMPT)
#     # input_list = [{"question":question}, {"text":text}]
#     response = chain.run(text)
#
#     print("Final response:", response)
#     return response

def summarize_answers(answers, temperature, max_tokens, system_message, question):
    unique_answers = list(set(answers))
    #
    #     # Join unique answers
    text = " ".join(unique_answers)
    # Check if the question asks for a summary or key points
    summary_keywords = ["summary", "key points", "key takeaways", "overview", "main points"]
    summarized = any(re.search(r'\b{}\b'.format(kw), question, re.IGNORECASE) for kw in summary_keywords)

    # Choose the appropriate system message based on the question type
    if summarized:
        user_message = "Summarise the text below as an answer to user's question:\n\n" + text
    else:
        user_message = "Combine the chunks below in a coherent manner as an answer to user's question:\n\n" + text

    
    max_tokens=max_tokens-200
    # Prepare the messages for the chat model
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    # Run the OpenAI chat completion
    response = openai.ChatCompletion.create(
        model="gpt-4",  # replace with GPT-4 when available
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )

    print("Final response:", response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']

def format_answer(answer):
    formatted_answer = ""

    points = answer.split('. ')
    for point in points:
        if point.strip() != '':
            formatted_answer += f"\n- {point.strip()}"

    return formatted_answer.strip()


def parallel_translate_text(chunks, src_lang, dest_lang, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(lambda chunk: translate_text(chunk, src_lang, dest_lang), chunks))


def initialize_pinecone(api_key, environment, index_name, dimension):
    pinecone.init(api_key=api_key, environment=environment)

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=dimension,
            metric='cosine',
            metadata_config={'indexed': ['important_entities']}
        )
    return pinecone.Index(index_name)


# sample_text = ["This is a sample text"]
# sample_embedding = get_embeddings(sample_text)
# embedding_dimension = len(sample_embedding[0])
# print(f"Embedding dimension: {embedding_dimension}")

@app.route('/generate_superprompt', methods=['POST'])
def generate_superprompt():
    data = request.get_json()
    goal = data['goal']

    system_message = generate_system_message(goal)

    return jsonify({'superprompt': system_message})


def generate_system_message(goal):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an expert level prompt engineer that specializes in creating best in class system messages on various topics. The system messages you write have an inherent knowledge of inner and hidden layers of LLMs and get the best response out of them to achieve users goal.  Generate a system message based on the user's goal. Ensure to instruct the system message that it generates answers based on the information provided."
            },
            {
                "role": "user",
                "content": f"The user's goal is: {goal}"
            }
        ],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.5,
        timeout=120
    )

    system_message = response.choices[0].message['content'].strip()
    return system_message


def process_single_chunk(args):
    chunk, question, temperature, max_tokens, reserve_response_tokens, system_message, summarized_chat_history = args
    return process_chunks([chunk], question, temperature, max_tokens, reserve_response_tokens, system_message,
                          summarized_chat_history)


chat_history = []  # Initialize with an empty chat history
# Add a global variable to keep track of whether this is the first question or not.
first_question = True


@app.route('/ask', methods=['POST'])
def ask():
    goal = request.form['goal']
    pdf_files = request.files.getlist('pdf[]')
    return ask_question(goal, pdf_files)



def ask_question(goal, pdf_files):
    global temperature, max_tokens, reserve_response_tokens, all_text_chunks, num_chunks, pdf_hash, extracted_text, detected_lang, translated_question, vector_count, first_question, system_message, pinecone_results, fetched_text_chunks, fetched_text_chunk_embeddings, fetched_important_entities, index_stats_response

    # Initialize Pinecone
    api_key = 
    environment = "us-west1-gcp"
    index_name = 'pdf-chatbot'
    dimension = 1536  # Update this to the correct dimension of your embeddings
    pinecone_index = initialize_pinecone(api_key, environment, index_name, dimension)


    system_message = generate_system_message(goal)
    #system_message = system_message
    print("Generated system message:", system_message)

    if not pdf_files:
        print('No PDF found')
        return jsonify({"error": "No PDF file(s) provided"}), 400

    question = request.form['question']
    print("Received question:", question)

    # Detect the language of the question
    question_lang = detect_language(question)
    print(f"Detected question language: {question_lang}")

    pdf_files = request.files.getlist('pdf[]')
    pdf_hashes = []

    question_length = len(question)

    for pdf_file in pdf_files:
        pdf_file.seek(0)  # Reset the file pointer
        pdf_hash = calculate_file_hash(pdf_file)
        pdf_hashes.append(pdf_hash)

        pdf_file.seek(0)  # Reset the file pointer again

    pdf_langs = [None] * len(pdf_files)
    pdf_texts = []
    all_text_chunks = []
    num_pages_total = 0

    question_tokens = len(question.split())

    # Calculate the temperature and max_tokens based on the number of pages and question length
    temperature, max_tokens = calculate_params(num_pages_total, question_length)

    # system_message = "You are a helpful assistant that specializes in writing reporter style summary about complex legal and non-legal texts, particularly for energy markets. If you don't find answer in the text, only respond 'No information available'"
    system_tokens = len(system_message.split())

    reserve_response_tokens = 100

    # truncated_text = truncate_text(pdf_text, max_tokens, question_tokens, system_tokens, reserve_response_tokens)
    # print(truncated_text)
    # Calculate the available tokens for each chunk
    max_chunk_tokens = max_tokens - system_tokens - question_tokens - reserve_response_tokens
    #print(max_chunk_tokens)

    # basic calculation end

    # Initialize lists to store data for each PDF file
    all_pdf_text_chunks = []
    all_pdf_important_entities = []
    all_pdf_text_chunk_embeddings = []

    # Iterate through each PDF file
    # Iterate through each PDF file
    for pdf_file, pdf_hash in zip(pdf_files, pdf_hashes):
        # Only process the PDF file if this is the first question
        if first_question:
            with app.test_client() as client:
                try:
                    response = client.post('/extract_text', content_type='multipart/form-data',
                                           data={'pdf': (pdf_file, pdf_file.filename)}).get_json()

                    # Check for errors in the response
                    if 'error' in response:
                        print(f"Error extracting text from {pdf_file.filename}: {response['error']}")
                        continue
                except requests.exceptions.HTTPError as e:
                    print(f"Exception occurred while extracting text from {pdf_file.filename}: {e}")
                    continue

            extracted_text, num_pages = response['text'], response['num_pages']
            pdf_texts.append(extracted_text)

            if not extracted_text.strip():
                return jsonify(
                    {"error": "The extracted text is empty. Please try again with a different PDF file."}), 400

            detected_lang = detect_language(extracted_text)
            pdf_langs.append(detected_lang)
            num_pages = response['num_pages']
            num_pages_total += num_pages  # Update the total number of pages

            # Translate the question to the language of the PDF content
            translated_question = translate_text(question, question_lang, detected_lang)
            print(f"Translated question: {translated_question}")

            print("File hash:", pdf_hash)
            index_stats_response = pinecone_index.describe_index_stats()
            # print(index_stats_response)

            cache_hit = False
            if pdf_hash in index_stats_response['namespaces']:
                vector_count = index_stats_response['namespaces'][pdf_hash]['vector_count']
                # print(vector_count)
                if vector_count != 0:
                    cache_hit = True
                    filtered_vector_ids = [f'{pdf_hash}_{i}' for i in range(vector_count)]
                    # print(filtered_vector_ids)
                    pinecone_results = pinecone_index.fetch(ids=filtered_vector_ids, namespace=pdf_hash)
                    # print(pinecone_results['vectors'].keys())
                else:
                    pinecone_results = {'vectors': {}}

            fetched_text_chunks, fetched_important_entities, fetched_text_chunk_embeddings = [], [], []
            if not cache_hit:
                print("Cache miss. Processing the PDF file...")
                keyword_cache.clear()

                # print("PDF_TEXT:", extracted_text[:100], "...")
                # with open('readme.txt', 'w') as f:
                # f.write(pdf_text)
                # Define your important entities
                # Extract important entities using the updated function
                important_entities = extract_entities_large_text(extracted_text)
                important_entities.extend([
                    'renewable energy', 'solar', 'wind', 'hydro', 'geothermal', 'power', 'power purchase agreement',
                    'ppa', 'investment', 'pricing', 'market dynamics', 'capacity', 'generation', 'energy storage',
                    'financing', 'policy', 'regulation', 'grid', 'interconnection', 'offtaker', 'developer',
                    'project finance',
                    'construction', 'operation', 'maintenance', 'warranty', 'guarantee', 'renewable portfolio standard'
                ])

                # Call the chunk_text function with max_chunk_tokens and reserve_response_tokens
                text_chunks = sliding_window_chunk(extracted_text, max_chunk_tokens)
                num_chunks = len(text_chunks)
                text_chunk_embeddings = get_embeddings(text_chunks)
                pinecone_data = []

                for i, embedding in enumerate(text_chunk_embeddings):
                    data_entry = {
                        'id': f'{pdf_hash}_{i}',
                        'values': embedding,
                        'metadata': {
                            'text_chunk': text_chunks[i],
                            'important_entities': important_entities
                        }
                    }
                    pinecone_data.append(data_entry)

                pinecone_index.upsert(vectors=pinecone_data, namespace=pdf_hash)
                print("New data added to Pinecone. Pinecone data:", pinecone_data)
                fetched_text_chunks = text_chunks
                fetched_important_entities = important_entities
                fetched_text_chunk_embeddings = text_chunk_embeddings
            else:
                print("Cache hit. Reusing cached data from Pinecone.")
                num_text_chunks = len(pinecone_results['vectors'])
                # print(range(num_text_chunks))
                pinecone_results = pinecone_index.fetch(ids=[f'{pdf_hash}_{i}' for i in range(num_text_chunks)],
                                                        namespace=pdf_hash)

                for i in range(num_text_chunks):
                    fetched_text_chunks.append(pinecone_results['vectors'][f'{pdf_hash}_{i}']['metadata']['text_chunk'])
                    fetched_important_entities.append(
                        pinecone_results['vectors'][f'{pdf_hash}_{i}']['metadata']['important_entities'])
                    fetched_text_chunk_embeddings.append(pinecone_results['vectors'][f'{pdf_hash}_{i}']['values'])

        # Add the data for the current PDF file to the corresponding lists
        all_pdf_text_chunks.append(fetched_text_chunks)
        all_pdf_important_entities.append(fetched_important_entities)
        all_pdf_text_chunk_embeddings.append(fetched_text_chunk_embeddings)

    # Get the question embedding
    question_embedding = get_embeddings([question])[0]

    # Combine all text_chunks, important_entities, and text_chunk_embeddings
    combined_text_chunks = list(itertools.chain.from_iterable(all_pdf_text_chunks))
    combined_important_entities = list(itertools.chain.from_iterable(all_pdf_important_entities))
    combined_text_chunk_embeddings = list(itertools.chain.from_iterable(all_pdf_text_chunk_embeddings))
    combined_pdf_langs = list(
        itertools.chain.from_iterable([[lang] * len(chunks) for lang, chunks in zip(pdf_langs, all_pdf_text_chunks)]))

    # Filter the top n relevant chunks
    n = 5  # Change this value to the desired number of top chunks
    text_chunks = filter_relevant_chunks_v3(combined_text_chunks, combined_important_entities, question_embedding,
                                            combined_text_chunk_embeddings, top_n=10)
    # Translate the PDF content to the language of the question
    # current_pdf_lang = pdf_lang
    translated_text_chunks = [translate_text(chunk, lang, question_lang) for chunk, lang in
                              zip(text_chunks, combined_pdf_langs)]

    chat_history_text = " ".join([f"Question: {q}\nAnswer: {a}" for q, a in chat_history])
    #print(chat_history_text)

    # Summarize the chat history
    if chat_history_text.strip():
        summarized_chat_history = summarize_answers([chat_history_text], temperature, max_tokens, system_message,
                                                    question)
    else:
        summarized_chat_history = ""

    chat_history_text_summary = chat_history_text + " overall summary of the chat prior to last question: " + summarized_chat_history

    # Include the summarized chat history in the system_message
    # system_message = f"{generate_system_message(goal)} {summarized_chat_history}"

    # Initialize with an empty chat history or update with previous chat history
    query = {"question": translated_question, "chat_history": chat_history_text_summary}
    processed_text_chunks = []

    # Create a list of arguments for the helper function
    args_list = [(chunk, query["question"], temperature, max_tokens, reserve_response_tokens, system_message,
                  query["chat_history"]) for chunk
                 in translated_text_chunks]

    # Parallelize the process_single_chunk function using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = []
        for args in args_list:
            future = executor.submit(process_single_chunk, args)
            futures.append(future)
            time.sleep(25)  # Sleep for 25 seconds

        for future in as_completed(futures):
            processed_text_chunks.append(future.result())

    flat_processed_text_chunks = [answer for chunk_answers in processed_text_chunks for answer in chunk_answers]
    coherent_answer = format_answer(
        summarize_answers(flat_processed_text_chunks, temperature, max_tokens, system_message, question))
    # Update the chat history with the current question and answer
    chat_history.insert(0,(question, coherent_answer))

    # Set first_question to False since we have processed the first question
    first_question = False

    return jsonify(answer=coherent_answer)


@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')


if __name__ == '__main__':
    test_openai_connection()
    app.run(host='127.0.0.1', port=7000, debug=True)
