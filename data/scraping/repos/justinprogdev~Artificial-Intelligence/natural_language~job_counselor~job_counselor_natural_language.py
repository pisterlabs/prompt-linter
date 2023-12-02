# Justin McClain Module 6 Natural Language Processing CS-664 Artificial Intelligence
# Required Dependency Installations:
# install textblob: pip install textblob
# install textblob corpa: python -m textblob.download_corpora
# install openai: pip install --pre openai
# install pandas: pip install pandas

import os
import json
import pandas as pd
from openai import OpenAI
from textblob import TextBlob
from job_data import JobData as job_data  # from file job_data.py
from nlp_rules import NlpRules as rules  # from file nlp_rules.py
from conversation_log import ConversationLog  # from file conversation_log.py

def analyze_sentiment(text):
    """Returns sentiment and intensity of sentiment if any"""
    if not text:
        return "Sentiment: N/A Intensity: N/A"
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    intensity = abs(blob.sentiment.subjectivity)
    return f"Sentiment: {sentiment} Intensity: {intensity}"


def main():
    # Get job dataframe for processing phrases with keywords/tokens
    # these keywords/tokens are for use with the vocational rules in nlp_rules.py
    df = pd.DataFrame(
        job_data().get_job_data(),
        columns=[
            "Job Title",
            "Education Requirement",
            "Job Description",
            "Available Salary Range",
        ],
    )

    # Conversation constraints and vocational rules
    # These rules are used to train the AI to respond to user input
    # and are the most important part of the AI's behavior.
    # please see nlp_rules.py for the complete sets of rules.
    conversation_rules = (
        rules().get_conversation_constraints() + " " + rules().get_vocational_rules() +" " + json.dumps(df.to_dict())
    )

    # Conversation log object to maintain conversation history
    conversation_log = ConversationLog(
        {"role": "system", "content": conversation_rules}
    )

    is_running = True
    user_prompt = (
        "NatLang: Welcome, I am NatLang, your Mid-life Career Change Virtual Advisor\n\n"
        "How can I help you today?\n\n>>>"
    )

    # Main conversation loop.
    while is_running:
        user_input = input(user_prompt)

        if user_input.lower() in ["exit", "bye", "goodbye"]:
            is_running = False

        # analyze sentiment of user input, so we can pass more info to the AI
        sentiment_output = analyze_sentiment(user_input)

        user_message = {
            "role": "user",
            "content": user_input + " | Sentiment: " + sentiment_output,
        }

        # add user input to conversation log, so it's not forgotten later
        conversation_log.add_message(user_message)

        # pass conversation log to OpenAI Completions API
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation_log.get_log(),
            temperature=0.8,
        )

        # add AI response to conversation log so it's not forgotten later
        ai_message = {
            "role": completion.choices[0].message.role,
            "content": completion.choices[0].message.content,
        }
        conversation_log.add_message(ai_message)

        # output our bot's response
        print(f"\nNatLang: {completion.choices[0].message.content}\n")
        user_prompt = ">>>"


if __name__ == "__main__":
    main()
