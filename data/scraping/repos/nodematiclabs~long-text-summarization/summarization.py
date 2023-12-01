import tiktoken

from langchain.chains import ConversationChain
from langchain.chat_models import ChatVertexAI
from langchain.memory import ConversationBufferMemory

encoder = tiktoken.get_encoding("cl100k_base")

def summarize_text(text, recursion_level=0):
    print("Recursion Level: ", recursion_level)

    # Split text into chunks of 4096 tokens
    texts = []
    text_tokens = encoder.encode(text)
    for i in range(0, len(text_tokens), 4096):
        texts.append(encoder.decode(text_tokens[i:i+4096]))

    llm = ChatVertexAI(
        max_output_tokens=1024
    )
    conversation = ConversationChain(
        llm=llm, memory=ConversationBufferMemory()
    )

    summarized_text = ""
    for i, text_segment in enumerate(texts):
        print("Text Segment: ", i)
        response = conversation.predict(input="Write a summary the following text:\n\n" + text_segment)
        summarized_text += response + "\n\n"
    if len(texts) == 1:
        return summarized_text
    else:
        return summarize_text(summarized_text, recursion_level=recursion_level+1)

with open("story.txt", "r") as f:
    text = f.read()
    print(summarize_text(text))