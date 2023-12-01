from llama_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, ServiceContext
from langchain import OpenAI

import os
open_api_key = os.getenv("OPENAI_API_KEY")

EMAIL_QUERIES =  {
    "recent_emails": "in:inbox -from(noreply* OR *mailer* OR *alert* OR *notification*)"
}

def get_prompt():
    QA_PROMPT_TMPL = (
        "You are a QA bot that helps answer questions about information in the context provided"
        "---------------------\n"
        "{context_str}"
        "-----------------------\n"
        "Given this information, answer the question: {query_str}\n"
    )
    QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
    return QA_PROMPT


def get_documents(email_query):
    GmailReader = download_loader('GmailReader')
    loader = GmailReader(query=email_query)
    documents = loader.load_data()
    print(f"loaded {len(documents)} documents")
    return documents


def get_index(documents):
    max_input_size = 4096
    num_output = 256
    max_chunk_overlap = 20
    chunk_size_limit = 600
    prompt_helper = PromptHelper(
        max_input_size=max_input_size,
        max_chunk_overlap=max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
        num_output=num_output,
    )
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0,
            model_name="text-ada-001",
            openai_api_key=openai_api_key,
            max_tokens=num_output,
        )
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        prompt_helper=prompt_helper,
        )
    try:
        index = GPTSimpleVectorIndex.load_from_disk("index.json")
    except FileNotFoundError:
        index = None
    print("index: ", index)
    if index is None:
        print("no index found, saving new index to disk")
        index = GPTSimpleVectorIndex.from_documents(
            documents,
            service_context=service_context
        )
        index.save_to_disk("index.json")
    return index

def get_email():
    documents = get_documents()
    index = get_index(documents)
    return "connected!"

def query_email(email_query):
    documents = get_documents(email_query)
    prompt = get_prompt()
    index = get_index(documents)
    return index.query(query, text_qa_template=prompt)

