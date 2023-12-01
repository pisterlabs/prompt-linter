import os
import random
import json
import pandas as pd
from textblob import TextBlob
from nrclex import NRCLex
import openai
import dotenv
import gradio as gr
from nltk.sentiment import SentimentIntensityAnalyzer

dotenv.load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize NLTK sentiment analyzer
sia = SentimentIntensityAnalyzer()

conversation_history = []
df = pd.DataFrame(columns=['user_message', 'bot_message', 'emotion_keywords'])

def get_bot_response(user_message, sentiment, previous_shadow):
    global conversation_history

    temperature = 0.8 if sentiment >= 0 else 1
    conversation_history.append({'role': 'user', 'content': user_message})
    conversation_history = conversation_history[-3:]

    highest_ranked_emotion = recall_emotion_keywords()[:1] or ['defensive']
    user_message += ' ' + highest_ranked_emotion[0]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role': 'system', 'content': f'You are Caulfield. You are very empathic and respond with deep feelings of {highest_ranked_emotion[0]}.'},
            {'role': 'system', 'content': f'You are to stay in character. Do not break character. Do not reply in a question. Never create lists. Answer questions as statements.'},
            {'role': 'system', 'content': f'Consider the previous criticism: {previous_shadow} before answering.'},
            *conversation_history
        ],
        max_tokens=300,
        n=1, 
        temperature=temperature,
    )

    bot_message = response['choices'][0]['message']['content']
    conversation_history.append({'role': 'assistant', 'content': bot_message})

    df.loc[len(df)] = [user_message, bot_message, '']
    update_emotion_keywords(bot_message)

    return bot_message


def extract_keywords(text):
    blob = TextBlob(text)
    keywords = blob.noun_phrases
    nrc_keywords = NRCLex(text).affect_frequencies
    emotion_keywords = [keyword for keyword, freq in nrc_keywords.items() if freq > 0]

    return keywords + emotion_keywords

def update_emotion_keywords(bot_message):
    global df

    df['emotion_keywords'] = ''

    for i, message in enumerate(conversation_history):
        text = message['content'] if message['role'] == 'assistant' else conversation_history[i - 1]['content']
        emotion_keywords = extract_keywords(text)
        df.at[i, 'emotion_keywords'] = ', '.join(emotion_keywords)

    df.at[len(df) - 1, 'emotion_keywords'] = ', '.join(extract_keywords(bot_message))

def recall_emotion_keywords():
    keywords = []
    for index, row in df.iterrows():
        if row['emotion_keywords']:
            keywords.extend(row['emotion_keywords'].split(', '))
    return [keyword for keyword in set(keywords)]

# Initialize conversation_info
conversation_info = {}

def chatbot(user_message):
    # Perform sentiment analysis on the user message
    sentiment_scores = sia.polarity_scores(user_message)
    sentiment = sentiment_scores['compound']

    # Get the previous shadow response
    global conversation_info
    previous_shadow = conversation_info.get('shadow', '')

    bot_message = get_bot_response(user_message, sentiment, previous_shadow)



    # Get the top keywords
    top_keywords = recall_emotion_keywords()[:3]

    # Random keyword selection
    random_keyword = random.choice(top_keywords)

    # Weighted keyword selection
    weights = [1, 2, 1]  # Example weights, adjust as needed
    weighted_keyword = random.choices(top_keywords, weights=weights, k=1)[0]

    # Subset keyword selection
    subset_size = 2  # Example subset size, adjust as needed
    subset_keywords = random.sample(top_keywords, subset_size)


    objective_prompt = f"Write an objective statement related to '{weighted_keyword}' and {user_message}"
    objective_response = openai.Completion.create(
        engine="text-davinci-003",
        frequency_penalty=0.8,
        prompt=objective_prompt,
        max_tokens=100
    )
    objective = objective_response.choices[0].text.strip()


    shadow_prompt = f"Considering {user_message}, write a harsh opposing critique of {objective_response} thoughts using {weighted_keyword},'{random_keyword}' as evidence."
    shadow_response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=shadow_prompt,
        #temperature=0.7,
        max_tokens=120,
        top_p=0.2,
        frequency_penalty=0.8,
        presence_penalty=0.6
    )
    shadow = shadow_response.choices[0].text.strip()

    # Format the response
    #bot_message += f"\n\nObjective:\n {objective}\n\nShadow:\n{shadow}"

    conversation_info = {
        'user_question': user_message,
        'sentiment': sentiment,
        'user_input': user_message,
        'top_keywords': top_keywords,
        'response': bot_message,
        'highest_rank_keyword': recall_emotion_keywords()[:1],
        'random_keyword': random_keyword,
        'weighted_keyword': weighted_keyword,
        'subset_keywords': subset_keywords,
        'objective': objective,
        'shadow': shadow
    }

    with open('conversation_info.json', 'a') as file:
        file.write(json.dumps(conversation_info) + '\n')

    return bot_message


iface = gr.Interface(fn=chatbot, inputs="text", outputs="text")
iface.launch(share=True)
