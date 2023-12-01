import os
import openai
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document


prompt_template = """Write a concise summary of the following:

{text}
It should be high level and summarize what the core topics
and actions are related to {user_id}. {user_id} is {user_name}

Start with 'Summary for {user_name}:'
CONCISE SUMMARY IN ENGLISH:
"""

def get_meta_summary(summaries, user):
    # Set the OpenAI API key.
    openai.api_key = os.getenv("OPENAI_API_KEY")

    llm = OpenAI(temperature=0.7, max_tokens=200)
    text_splitter = CharacterTextSplitter(chunk_size=200)

    user_id = user.id
    user_name = user.name
    user_text = " ".join(summaries)

    texts = text_splitter.split_text(user_text)
    docs = [Document(page_content=t) for t in texts]

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text","user_id", "user_name"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    summary = chain.run(input_documents=docs, user_id=user_id, user_name=user_name)

    # Return the summary.
    return summary['output_text']