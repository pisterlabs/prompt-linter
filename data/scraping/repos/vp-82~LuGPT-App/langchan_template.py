# %%
import os
import re
from pprint import pprint

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.conversational_retrieval.prompts import \
    CONDENSE_QUESTION_PROMPT
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Milvus

# %%
load_dotenv()  # take environment variables from .env.
openai_api_key = os.getenv('OPENAI_API_KEY')
milvus_api_key = os.getenv('MILVUS_API_KEY')

connection_args={
        "uri": "https://in03-5052868020ac71b.api.gcp-us-west1.zillizcloud.com",
        "user": "vaclav@pechtor.ch",
        "token": milvus_api_key,
        "secure": True
    }

# %%
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
milvus = Milvus(
    embedding_function=embeddings,
    collection_name="LuGPT",
    connection_args=connection_args,
)
chat_history = []

# %%
prompt_template="""Angesichts der folgenden Konversation und einer anschließenden Frage, formulieren Sie die Nachfrage so um, dass sie als eigenständige Frage gestellt werden kann.
Alle Ausgaben müssen in Deutsch sein.
Wenn Sie die Antwort nicht kennen, sagen Sie einfach, dass Sie es nicht wissen, versuchen Sie nicht, eine Antwort zu erfinden.

Chatverlauf:
{chat_history}
Nachfrage: {question}
Eigenständige Frage:

"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["chat_history", "question"]
)

# %%
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k-0613')
question_generator = LLMChain(llm=llm,
                              prompt=PROMPT,
                            )
doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")

# %%
chain = ConversationalRetrievalChain(
    retriever=milvus.as_retriever(),
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
)

# %%
query = "Was macht die Dienststelle Informatik"

# %%
chat_history = []
result = chain({"question": query, "chat_history": chat_history})

# %%
pprint(result)

# %%
chat_history = [(query, result["answer"])]

# %%
query = "Welche Projekte macht sie?"
result = chain({"question": query, "chat_history": chat_history})

# %%
pprint(result)

# %%
def process_output(output):
    # Check if 'SOURCES: \n' is in the output
    if 'SOURCES:' in output['answer']:
        # Split the answer into the main text and the sources
        answer, raw_sources = output['answer'].split('SOURCES:', 1)

        # Split the raw sources into a list of sources, and remove any leading or trailing whitespaces
        raw_sources_list = [source.strip() for source in raw_sources.split('- ') if source.strip()]

        # Process each source to turn it back into a valid URL
        sources = []
        for raw_source in raw_sources_list:
            if raw_source:  # Ignore empty strings
                # Remove the ending '.txt' and replace '__' with '/'
                valid_url = 'https://' + raw_source.replace('__', '/').rstrip('.txt\n')
                sources.append(valid_url)
    else:
        # If there are no sources, return the answer as is and an empty list for sources
        answer = output['answer']
        sources = []

    # Join the sources list into a single string with each source separated by a whitespace
    sources = ' '.join(sources)
        
    return answer, sources

# %%
antwort, quellen = process_output(result)

# %%
pprint(antwort)
pprint(quellen)

# %%



