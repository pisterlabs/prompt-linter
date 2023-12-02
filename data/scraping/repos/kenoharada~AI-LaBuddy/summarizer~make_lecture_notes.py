import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import TokenTextSplitter


def split_chunk_by_token_nums(text):
    splitter = TokenTextSplitter(encoding_name='cl100k_base', chunk_size=3072, chunk_overlap=0)
    return splitter.split_text(text)

def summarize_to_notes(text, title):
    chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chat.openai_api_key = os.environ['OPENAI_API_KEY']
    prompt = f"""
    ###
    {text}
    ###
    The above ##-encircled area is the important points of the presentation given under the title {title}. Please summarize points in 3-5 sentences.
    """
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt),
    ]
    response = chat(messages)
    # print(response)
    print(response.content)
    return response.content


if __name__ == '__main__':
    text = open('/Users/haradakeno/localdev/AI-LaBuddy/summarizer/CS50W - Lecture 0 - HTML and CSS/summary.txt').read()
    text_splits = split_chunk_by_token_nums(text)
    notes = ''
    for text_split in text_splits:
        note = summarize_to_notes(text_split, 'CS50W - Lecture 0 - HTML and CSS')
        notes += note
        notes += '\n'
    with open('/Users/haradakeno/localdev/AI-LaBuddy/summarizer/CS50W - Lecture 0 - HTML and CSS/notes.txt', 'w') as f:
        f.write(note)
        f.write('\n')