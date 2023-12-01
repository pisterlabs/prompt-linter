from openai.embeddings_utils import cosine_similarity, get_embedding
import openai
openai.api_key ='INSERT API KEY HERE'

labels = ['Positive', 'Negative']
label_embeddings = [get_embedding(label, engine="text-similarity-davinci-001") for label in labels]
sentence = [get_embedding("Joy", engine="text-similarity-davinci-001")]

def label_score(input, label_embeddings):
   return cosine_similarity(input, label_embeddings[1]) - cosine_similarity(input, label_embeddings[0])

prediction = 'Negative' if label_score(sentence, label_embeddings) > 0 else 'Positive'
print(prediction)