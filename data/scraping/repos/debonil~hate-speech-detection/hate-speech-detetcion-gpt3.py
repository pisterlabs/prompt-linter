# %%
#!pip install python-dotenv
#!pip install openai

# %%
import openai
import nltk

# Authenticate with OpenAI's GPT-3 API
openai.api_key = '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'

# %%
import openai
import re

# Set your API key
#openai.api_key = "<YOUR_API_KEY>"

# Define the function to detect hate speech
def detect_hate_speech(sentence):
    # Classify the sentence as either hate speech or not hate speech using GPT-3
    prompt = f"Classify the following sentence as hate speech or not hate speech:\n{sentence}\n---\n"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=128, # Increase max_tokens to include the entire classification
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    classification = response.choices[0].text.strip()
    print(classification)
    if classification == "This sentence is hate speech." or  classification == "This is hate speech." or classification =="This sentence is classified as hate speech." or classification =="Hate speech." or classification =="hate speech":
        # Extract the hate speech words using regular expressions
        #hate_speech_words = re.findall(r"\b(?!not\b)\w+", sentence.lower())
        hate_speech_words =["nigger", "nigga", "negro", "muslim", "Black", "whore", "fuck", "cuck", "terrorist", "asshole", "cunt", "fucker", "kill", "bomb", "shoot", "commies", "leftist", "trump", "white", "blonde", "dead"]
        num_words=0
        for word in sentence.split():
            if word.lower() in hate_speech_words:
                num_words +=1
        
        #num_words = len(hate_speech_words)
        total_words = len(sentence.split())
        percentage = num_words / total_words * 100
        return hate_speech_words, percentage
    else:
        return None

# Get input sentence from user
sentence = input("Enter a sentence: ")

# Detect hate speech in the input sentence using the detect_hate_speech function
result = detect_hate_speech(sentence)

# Output the results
if result:
    print(f"The input sentence contains hate speech.")
    #print(f"Hate speech words: {result[0]}")
    print(f"Percentage of hate speech: {result[1]:.2f}%")
else:
    print(f"The input sentence does not contain hate speech.")



