# IMPORTS
import os
import spacy
import openai
from dotenv import load_dotenv
from Model.AI.Role import kwords
# from gtts import gTTS 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# TO INITIATE (.env) FILE
load_dotenv()
# IT CALLS THE ROLE MODELS AND ITS COMPONENTS
rwords = kwords()

# IT CALLS THE API KEY
apikey = os.getenv('API_KEY')
openai.api_key = apikey

# Load a spaCy model for text processing
nlp = spacy.load("en_core_web_sm")

# Initialize a dictionary to store user-specific information
user_info = {}

def say(text):
    # Split the text into words and count them
    word_count = len(text.split())
    if word_count <= 100:
        print("VidyaAI: "+text)
        os.system(f'say "{text}"')
    else:
        print("VidyaAI: "+text)
            

# Define a dictionary of roles and their associated keywords
def categorize_role(prompt):
    # Tokenize the user's input prompt
    input_tokens = [token.text for token in nlp(prompt.lower())]

    # Create TF-IDF vectorizers for input tokens and role keywords
    tfidf_vectorizer = TfidfVectorizer()
    input_vector = tfidf_vectorizer.fit_transform([" ".join(input_tokens)])
    role_similarities = {}
    for role, keywords in rwords.items():
        # Convert role keywords into a single string for TF-IDF vectorization
        keyword_text = " ".join(keywords)
        keyword_vector = tfidf_vectorizer.transform([keyword_text])
        # Calculate cosine similarity between input and role keywords
        similarity_matrix = cosine_similarity(input_vector, keyword_vector)
        role_similarities[role] = similarity_matrix[0][0]

    # Categorize the role based on the highest similarity
    categorized_role = max(role_similarities, key=role_similarities.get)
    return categorized_role

# CHAT BOT FUNCTION TO CALL API AND GET RESPONSE
def Chat(role, temperature, top_p, prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Your name is VidyaAI and You will serve as {role} assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=500,
        frequency_penalty=0.3,
        presence_penalty=0.3)
    return response["choices"][0]["message"]["content"].strip()

# THE MAIN AI FUNCTION FOR PROMPT INPUT AND ROLE ASSIGNMENT
def AI(prompt):
    role = categorize_role(prompt)
    # CONDITIONAL STATEMENT TO FIND OUT THE ROLE
    if role == "General":
        aspect = ["General ChatBot", 0, 0]
    elif role == "Mathematics":
        aspect = ["Mathematics", 0, 0.8]
    elif role == "Data Analysis":
        aspect = ["Data Analysis", 0.2, 0.1]
    elif role == "Code Explainer":
        aspect = ["Code Explainer", 0.6, 0.7]
    elif role == "Code Generating":
        aspect = ["Code Generating", 0.2, 0.1]
    elif role == "Comment Generating":
        aspect = ["Comment Generating", 0.3, 0.2]
    elif role == "Professional ChatBot":
        aspect = ["Professional ChatBot", 0.5, 0.5]
    elif role == "Creative Story Writing":
        aspect = ["Creative Story Writing", 0.7, 0.8]
    role, temperature, top_p = aspect

    # Check if the prompt contains a question about the user's name
    if "my name is" in prompt.lower():
        # Extract the user's name
        user_name = prompt.lower().split("my name is")[1].strip()
        # Store the user's name in the user_info dictionary
        user_info["name"] = user_name

    # Check if the prompt is asking for the user's name
    if "what is my name" in prompt.lower():
        # Retrieve the user's name from the user_info dictionary
        user_name = user_info.get("name", "I don't know your name.")
        # Generate a response with the user's name
        response = f"Your name is {user_name}."
        say(response)
        return response

    response = Chat(role, temperature, top_p, prompt=prompt)
    say(response)
    return response
