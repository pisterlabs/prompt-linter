import os
import openai
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from fastapi_app.chatbot.prompt import prompt_template_summary, prompt_template_question, PROMPT_REGULAR
from fastapi_app.chatbot.count_costs import num_tokens_from_messages
#from chatbot.prompt import prompt_template_summary, prompt_template_question, PROMPT_REGULAR
#from chatbot.count_costs import num_tokens_from_messages

OPENAI_API_KEY = "your_key"
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROMPT_QUESTION = PromptTemplate(template=prompt_template_question, input_variables=["text"])
PROMPT_SUMMARY = PromptTemplate(template=prompt_template_summary, input_variables=["text"])


def count_tokens(text, prompt=PROMPT_QUESTION):
    messages = [{"role": "system", "content": f"{prompt} \'{text}\'"}]
    num_tokens = num_tokens_from_messages(messages)
    print("\033[96mNumber of tokens:", num_tokens, '\033[0m')
    return num_tokens


def get_summary(docs, api_key=OPENAI_API_KEY, prompt=PROMPT_SUMMARY, is_large=False):
    summary_chain = load_summarize_chain(OpenAI(temperature=0, openai_api_key=api_key),
                                         chain_type="map_reduce",
                                         return_intermediate_steps=False,
                                         map_prompt=prompt,
                                         combine_prompt=prompt)
    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)

    if is_large:
        result = summary_chain({"input_documents": docs}, return_only_outputs=True)
    else:
        result = summarize_document_chain.run(docs)
    print(result)
    return result


def get_answer(document, question=PROMPT_QUESTION, api_key=OPENAI_API_KEY,
               chain_type="refine", temperature=0.02):
    qa_chain = load_qa_chain(OpenAI(temperature=temperature, openai_api_key=api_key),
                             chain_type=chain_type)
    qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

    answer = qa_document_chain.run(input_document=document, question=question)
    print(answer)
    return answer


def get_answer_simple(question, prompt=None, api_key=OPENAI_API_KEY):
    openai.api_key = api_key
    from openai.error import InvalidRequestError
    prompt = prompt or PROMPT_REGULAR
    messages = [{"role": "system", "content": f"{prompt} \'{question}\'"}]
    num_tokens = num_tokens_from_messages(messages)
    print("\033[92mNumber of tokens:", num_tokens, '\033[0m')

    print("Number of tokens is more than maximum for GPT-3. GPT-4 will be used") if num_tokens >= 4097 else None
    model_type = "gpt-3.5-turbo" if num_tokens < 4097 else "gpt-4"

    try:
        completion = openai.ChatCompletion.create(
            model=model_type,
            messages=messages,
            temperature=0,
            max_tokens=2000,
        )
        summary = completion['choices'][0]['message']['content'].strip()

        print(f"\n\33[90mAnswer: {summary}\33[0m")
        return {"answer": summary, "sources": {"OpenAI": {"href": "https://chat.openai.com/", "name": "OpenAI GPT-3.5 Turbo"}}}

    except InvalidRequestError as e:
        print(
            "Currently, OpenAI uses a waitlist system for granting access to the GPT-4 API.")


def _split_text(text, chunk_size=512, verbose=True):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"],
                                                   chunk_size=chunk_size,
                                                   chunk_overlap=500)
    docs = text_splitter.create_documents([text, ])

    if verbose:
        print("Number of documents:", len(docs))
        for i, doc in enumerate(docs):
            print(f"[doc {i}] Number of tokens:", count_tokens(doc.page_content))

    return docs


def _fix_answer(text):
    text = text.replace("AI language model", "AI assistant")
    text = text.replace("AI model", "AI assistant")
    text = text.replace("AI system", "AI assistant")
    return text


if __name__ == "__main__":
    from fastapi_app.chatbot.translation import translate_ruen, translate_enru

    with open("task.txt", "r", encoding="utf-8") as f:
        data = f.read()

    # count_tokens(data, prompt=PROMPT_SUMMARY)
    # docs = _split_text(data, chunk_size=2000)
    # document = Document(page_content=data)
    # get_summary(docs, is_large=True)

    document = "Основные нормы законодательства Москвы про ведение бизнеса."
    d_en = translate_ruen(document)
    task = "system: You are a Russian business expert. Please provide a concise answer to the question below."
    answer = get_answer_simple(question=d_en, prompt=task, api_key=OPENAI_API_KEY)
    d_ru = translate_enru(answer["answer"])
    print(d_ru)
