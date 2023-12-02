from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai_api_key = os.getenv("openai_api_key")
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
llm3 = ChatOpenAI(temperature=0,
                  openai_api_key=openai_api_key,
                  max_tokens=1000,
                  )

def getReply(text, final_prompt="what is this story? do you think it is a good story?", compress_prompt="please compress the text and output key information"):
    text = compressText(text, compress_prompt, final_prompt)
    return finalPrompt(final_prompt, text)


def compressText(text, compress_prompt, final_prompt):
    condition = llm.get_num_tokens(text + final_prompt) > 3800
    while condition:
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t", " ", ".", ","], chunk_size=10000,
                                                       chunk_overlap=3000)
        docs = text_splitter.create_documents([text])
        num_documents = len(docs)
        print(f"Now our text is split up into {num_documents} documents")
        selected_docs = docs

        map_prompt = compress_prompt + "\n" + "```{text}```"

        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
        map_chain = load_summarize_chain(llm=llm3,
                                         chain_type="stuff",
                                         prompt=map_prompt_template)
        selected_docs = [doc for doc in selected_docs]
        summary_list = []

        # Loop through a range of the length
        for i, doc in enumerate(selected_docs):
            # Go get a summary of the chunk
            chunk_summary = map_chain.run([doc])
            # Append that summary to list
            summary_list.append(chunk_summary)

        summaries = "\n".join(summary_list)
        text = summaries
        condition = llm.get_num_tokens(summaries + final_prompt) > 3800
    return text

def finalPrompt(final_prompt, text):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    prompt = f"{final_prompt}\n```{{text}}```\ntext = {text}"
    output = llm(prompt)
    return output
