import os
import wandb
from langchain.llms import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import pandas as pd
import pickle 
import faiss

from utils import create_html, color_start, color_end
from prompt import prompt_template, system_template

from types import SimpleNamespace

cfg = SimpleNamespace(
    TEMPERATURE = 0,
    PROJECT = "wandb_docs_bot",
    INDEX_ARTIFACT = "darek/wandb_docs_bot/faiss_store:v2",
    PROMPT_TEMPLATE = prompt_template,
    MODEL = "chatGPT",
)

def load_vectostore():
    artifact = wandb.use_artifact(cfg.INDEX_ARTIFACT, type='search_index')
    artifact_dir = artifact.download()
    index = faiss.read_index(artifact_dir + "/docs.index")
    with open(artifact_dir + "/faiss_store.pkl", "rb") as f:
        store = pickle.load(f)
    store.index = index
    return store


def load_prompt():
    prompt = PromptTemplate(input_variables=["question", "summaries"],
                          template=cfg.PROMPT_TEMPLATE)
    return prompt

def load_chain(openai_api_key):
    vectorstore = load_vectostore()
    prompt = load_prompt()
    chain = VectorDBQAWithSourcesChain.from_llm(
        llm=OpenAI(temperature=cfg.TEMPERATURE, openai_api_key=openai_api_key),
        vectorstore=vectorstore,
        combine_prompt=prompt)
    return chain, prompt

def get_answer(question, chain):
    if chain is not None:
        chain.return_source_documents = True
        result = chain(
          {
            "question": question,
          },
          return_only_outputs=False,
        )
        return result['answer'], result["source_documents"], result['sources']

    
openai_api_key = os.getenv("OPENAI_API_KEY")
if len(openai_api_key) < 10:
    raise ValueError("Set OPENAI_API_KEY environment variable")

run = wandb.init(project=cfg.PROJECT, config=cfg)

eval_table = wandb.Table(columns=["question", "answer", "target", "prompt", "docs"])

df = pd.read_csv('llm_eval_set.csv', header=1).dropna()

vectorstore = load_vectostore()

for question, target in zip(df.Question, df.Answer):
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)    
    chain_type_kwargs = {"prompt": prompt}
    chain = VectorDBQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", vectorstore=vectorstore, chain_type_kwargs=chain_type_kwargs)    
    answer, docs, sources = get_answer(question, chain)
    docs_string = '\n\n'.join([color_start + d.metadata['source'] + ':\n' + color_end + d.page_content for d in docs])
    docs_html = wandb.Html(create_html(docs_string))
    answer_html = wandb.Html(create_html(answer))
    prompt_html = wandb.Html(create_html(system_template))
    question_html = wandb.Html(create_html(question))
    target_html = wandb.Html(create_html(target))
    eval_table.add_data(question_html, answer_html, target_html, prompt_html, docs_html)

wandb.log({'eval_table': eval_table})
run.finish()
print('done')

