import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# # Summaries of Short Text
# from langchain.llms import OpenAI
# from langchain import PromptTemplate

# llm = OpenAI(temperature=0, model_name = 'text-davinci-003', openai_api_key=openai_api_key)

# template = """
# %INSTRUCTIONS:
# Please summarise the following piece of text.
# Respond in a manner that a 5 year old would understand.

# %TEXT:
# {text}
# """

# # Create a LangChain prompt template that we can insert values to later
# prompt = PromptTemplate(
#     input_variables=["text"],
#     template=template
# )

# confusing_text = """
# For the next 130 years, debate raged.
# Some scientists called Prototaxites a lichen, others a fungus, and still others clung to the notion that it was some kind of tree.
# “The problem is that when you look up close at the anatomy, it’s evocative of a lot of different things, but it’s diagnostic of nothing,” says Boyce, an associate professor in geophysical sciences and the Committee on Evolutionary Biology.
# “And it’s so damn big that when whenever someone says it’s something, everyone else’s hackles get up: ‘How could you have a lichen 20 feet tall?’”
# """

# # print ("------- Prompt Begin -------")

# final_prompt = prompt.format(text=confusing_text)
# # print(final_prompt)

# # print ("------- Prompt End -------")

# output = llm(final_prompt)
# print (output)


# Summaries of Longer Text
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

with open('data/pgessay.txt', 'r') as file:
    text = file.read()

# Printing the first 285 characters as a preview
# print (text[:285])

num_tokens = llm.get_num_tokens(text)
# print (f"There are {num_tokens} tokens in your file")

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=5000, chunk_overlap=350)
docs = text_splitter.create_documents([text])

print (f"You now have {len(docs)} docs instead of 1 piece of text")

# Get your chain ready to use
chain = load_summarize_chain(llm=llm, chain_type='map_reduce')

# Use it. This will run through the 4 documents, summarise the chunks, then get a summary of the summary.
output = chain.run(docs)
print (output)
