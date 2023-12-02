import openai
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
openai.api_key = os.getenv('OPENAI_API_KEY')


# Function to get OpenAI embedding
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']



# Generate embeddings
def generate_embeddings(script, urls):
    # Load classifier words from urls dictionary
    classifier_words = list(urls.keys())
    
    # Get embeddings for classifier words
    classifier_embeddings = {word: get_embedding(word) for word in classifier_words}
    
    # Get embeddings for script
    script_embeddings = {key: get_embedding(sentence) for key, sentence in script.items()}
    
    # Dictionary to store the most relevant classifier for each frame
    most_relevant_classifiers = {}
    
    # Find the most relevant classifier for each frame
    for script_key, script_embedding in script_embeddings.items():
        max_similarity = -1
        relevant_classifier = ""
        for classifier_key, classifier_embedding in classifier_embeddings.items():
            similarity = cosine_similarity([np.array(script_embedding)], [np.array(classifier_embedding)])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                relevant_classifier = classifier_key
        most_relevant_classifiers[script_key] = urls[relevant_classifier]

    return list(script.values()),list(most_relevant_classifiers.values())


