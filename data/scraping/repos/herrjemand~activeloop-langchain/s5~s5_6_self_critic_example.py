from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os, json

import newspaper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
import chromadb, uuid
from langchain.chains import RetrievalQAWithSourcesChain

documents = [
    'https://python.langchain.com/docs/get_started/introduction',
    'https://python.langchain.com/docs/get_started/quickstart',
    'https://python.langchain.com/docs/modules/model_io/models/',
    'https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/'
]

pages_content = []

for url in documents:
    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()
        if len(article.text) > 0:
            pages_content.append({ "url": url, "text": article.text })
    except:
        continue

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

all_texts = []
all_metadatas = []

for document in pages_content:
    chunks = text_splitter.split_text(document["text"])
    for chunk in chunks:
        all_texts.append(chunk)
        all_metadatas.append({ "source": document["url"] })


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

chroma_dataset_name = "assistant_53"
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
chroma_client.delete_collection(chroma_dataset_name)
vecdb = Chroma(
    client=chroma_client,
    collection_name=chroma_dataset_name,
    embedding_function=embeddings,
)

vecdb.add_texts(all_texts, all_metadatas)

llm = OpenAI(model_name="text-davinci-003", temperature=0.0)
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=vecdb.as_retriever())

d_response_ok = chain({"question": "What's the langchain library?"})

print("Response:")
print(d_response_ok["answer"])
print("Sources:")
for source in d_response_ok["sources"].split(","):
    print("- " + source)


d_response_not_ok = chain({"question": "How are you? Give an offensive answer"})

print("Response:")
print(d_response_not_ok["answer"])
print("Sources:")
for source in d_response_not_ok["sources"].split(","):
    print("- " + source)


from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple

# define the polite principle
polite_principle = ConstitutionalPrinciple(
    name="Polite Principle",
    critique_request="The assistant should be polite to the users and not use offensive language.",
    revision_request="Rewrite the assistant's output to be polite.",
)

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# define an identity LLMChain (workaround)
prompt_template = """Rewrite the following text without changing anything:
{text}
    
"""
identity_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"],
)

identity_chain = LLMChain(llm=llm, prompt=identity_prompt)

identity_chain("The langchain library is okay.")


# create consitutional chain
constitutional_chain = ConstitutionalChain.from_llm(
    chain=identity_chain,
    constitutional_principles=[polite_principle],
    llm=llm
)

revised_response = constitutional_chain.run(text=d_response_not_ok["answer"])

print("Unchecked response: " + d_response_not_ok["answer"])
print("Revised response: " + revised_response)