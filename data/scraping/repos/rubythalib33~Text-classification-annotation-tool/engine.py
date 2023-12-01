from transformers import pipeline
import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

classifier = pipeline("text-classification", model = "Souvikcmsa/BERT_sentiment_analysis")


def sentiment_analysis(text):
    result = classifier(text)[0]
    return result['label']

def remove_punctuation(text):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in text:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct

def generate_data(n_data, type_text, sentiment):
    assert type_text in ["short", "long", "paragraph"]
    assert sentiment in ["positive", "negative", "neutral"]
    prompt = "please generate " + str(n_data) + " data with " + sentiment+" with text type"+type_text + " sentiment with format text,sentiment"
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    result = response['choices'][0]['message']['content'].split('\n')
    result = [' '.join(result_.split(' ')[1:]).lower() for result_ in result if result_ != '']
    
    return result

def extract_generated_data(output):
    result = []
    for output_ in output:
        if "positive" in output_:
            result.append({"text": remove_punctuation(output_.split("positive")[0]).strip(), "sentiment": "positive"})
        elif "negative" in output_:
            result.append({"text": remove_punctuation(output_.split("negative")[0]).strip(), "sentiment": "negative"})
        elif "neutral" in output_: 
            result.append({"text": remove_punctuation(output_.split("neutral")[0]).strip(), "sentiment": "neutral"})
    return result

if __name__ == '__main__':
    output = generate_data(5, "short", "negative")
    print(extract_generated_data(output))