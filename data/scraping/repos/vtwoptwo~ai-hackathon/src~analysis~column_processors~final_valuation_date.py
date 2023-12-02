from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import re

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.5)

re_date_pattern = r"((0[1-9]|[12][0-9]|3[01])\/(0[1-9]|1[0-2])\/([0-9]{4}))"

synonyms = ["Final Valuation Date", "Redemption Valuation Date", "Determination Date"]

faiss_prompt = f"What is the {', '.join(synonyms)} of this term sheet?"

prompt = PromptTemplate.from_template(
    "Take a deep breath and relax. Think step by step."
    "I have the following  document of a term sheet"
    "I need to find the Final Valuation Date (which is the last day that the product trades or is valid.)"
    "The format of the Final Valuation Date is usually in dd/mm/yyyy, mm/dd/yyyy, yyyy/mm/dd, or in natural language using the name of the month."
    "It is also sometimes called `Determination Date` or `Redemption Valuation Date`"
    "It is really important that you get this right, because my life depends on it!\n"
    "Example:"
    "Context: Final Valuation Date 3rd July 2020"
    "Final Valuation Date: 03/07/2020"
    "Example:"
    "Context: Final Valuation Date 2018.03.01"
    "Final Valuation Date: 01/03/2018"
    "I am going to give you a chunk of text, and you need to tell me which one is the Final Valuation Date of the document\n\n"
    "You must return it in the correct date format: dd/mm/yyyy and only return this value"
    "Context:{context}\n"
    "Final Valuation Date: <your answer here>"
)


def open_json_and_search_key_value_pairs(
    document_path: str, synonyms: list
) -> tuple[str, str, str, dict]:
    """
    opens the given json and searches keys for the list of synonyms (different names the feature can have)

    returns tuple of found_date, matched_synonym, full_text, key_value_pairs.
    If date is not found, found_date and matched_synonym will be None
    """

    with open(document_path, "r") as f:
        loaded_json = json.load(f)

    json_key_value_pairs = loaded_json["key-value_pairs"]
    full_text = loaded_json["full_text"]

    date = None
    matched_synonym = None

    for i in synonyms:
        date = json_key_value_pairs.get(i)

        if date:
            date = date[0]
            matched_synonym = i

            break

    return date, matched_synonym, full_text, json_key_value_pairs


def get_relevant_documents(full_text: str, faiss_prompt: str) -> list:
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separator=" "
    )
    texts = text_splitter.split_text(full_text)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(texts, embeddings)

    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(faiss_prompt)

    return docs


def form_prompt_context_and_call_llm(
    docs: list,
    prompt: str,
    found_date: str = None,
    matched_synonym: str = None,
    json_key_value_pairs: dict = None,
):
    full_context = ""

    for doc in docs:
        full_context += "\n" + doc.page_content

    if found_date:
        assert (
            matched_synonym
        ), "If found date is provided, matched_synonym is compulsory"
        assert (
            json_key_value_pairs
        ), "If found date is provided, json_key_value_pairs is compulsory"

        found_date_confidence = json_key_value_pairs[matched_synonym][1]

        full_context += (
            "\n"
            + f" I also did an OCR anaysis and found Final Valuation Date: {found_date} with a confidence of {found_date_confidence}, take this into consideration."
        )

    final_prompt = prompt.format(context=full_context)
    result = llm.invoke(final_prompt)

    return result


def get_final_valuation_date(document_path: str) -> str:
    (
        found_date,
        matched_synonym,
        full_text,
        json_key_value_pairs,
    ) = open_json_and_search_key_value_pairs(document_path, synonyms)
    docs = get_relevant_documents(full_text, faiss_prompt)
    result = form_prompt_context_and_call_llm(
        docs, prompt, found_date, matched_synonym, json_key_value_pairs
    )
    # print(result)
    matches = re.findall(re_date_pattern, result)

    if matches:
        final_val_date = matches[0][0]
        return final_val_date

    return None


# document_path = 'data_0611/OCR_output/XS2444480540.json'
# final_val_date = get_final_valuation_date(document_path)
# print(final_val_date)
