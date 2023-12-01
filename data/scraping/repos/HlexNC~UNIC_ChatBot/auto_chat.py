import os
import openai
import pandas as pd
from gpt_3_fine_tuning.semantic_search import search_takeaways, merge_dataframes_3
from gpt_3_fine_tuning.models import get_chat_completion, get_completion
from dotenv import load_dotenv

messages_data = []
messages = [
    {"role": "system", "content": "You are an AI 'semantic search' prompt generator for the University of Nicosia ("
                                  "UNIC). \n\nExample:\nUser input: What is UNIC?\nSemantic search prompt: University "
                                  "of Nicosia (UNIC) Overview \n\nRead the user message and the message history and "
                                  "determine what the prompt should be. Keep the prompt short and to the point. "},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "'Greetings and introductory messages'"},
    {"role": "user", "content": "What can you tell me about UNIC?"},
    {"role": "assistant", "content": "'University of Nicosia (UNIC) Overview'"},
    {"role": "user", "content": "What can you tell me about the Business school?"},
    {"role": "assistant", "content": "'Programs offered at the University of Nicosia (UNIC) Business School.'"},
    {"role": "user", "content": "Tell me more about it"},
    {"role": "assistant", "content": "'University of Nicosia (UNIC) Business School Tuition'"},
    # {"role": "user", "content": "What can you tell me about the Computer Science program at UNIC?"},
    # {"role": "assistant", "content": "'The University of Nicosia (UNIC) Computer Science Program Overview'"},
    # {"role": "user", "content": "What else?"},
    # {"role": "assistant", "content": "'The University of Nicosia (UNIC) Computer Science Program Curriculum'"},
    # {"role": "user", "content": "More info"},
    # {"role": "assistant", "content": "'The University of Nicosia (UNIC) Computer Science Program Admission Requirements'"},
    # {"role": "user", "content": "Thank you, bye"},
    # {"role": "assistant", "content": "'Goodbye and closing remarks.'"},
]


def chat(df):
    while True:
        user_message = input("User: ")
        if user_message == "quit":
            break
        messages_data.append({"role": "user", "content": user_message})
        messages.append({"role": "user", "content": user_message})
        response = get_chat_completion(messages)
        messages.append({"role": "assistant", "content": response})
        # print("> Looking for: ", response)
        data_retrival_result = search_takeaways(response, df, n=3)
        prompt = f'Data:\n{data_retrival_result}\n\nUsing the data reply to the following message:\n\n' \
                 f'Message: {user_message}\n\n' \
                 f'You are an AI assistant for the University of Nicosia. You provide answers to user inquiries using ' \
                 f'the data provided. Use a friendly and professional tone.\nUse data only if necessary.\n\nAnswer:'
        answer = get_completion(prompt)
        print("UNICorn: ", answer)
        messages_data.append({"role": "assistant", "content": answer})


if __name__ == '__main__':
    df = pd.read_json('data/merged_embedding.jsonl', lines=True)
    chat(df)
