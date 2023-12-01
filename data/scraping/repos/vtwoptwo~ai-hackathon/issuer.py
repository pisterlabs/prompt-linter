from langchain import OpenAI, PromptTemplate
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import re
from .helpers.utils import instruction_response

load_dotenv()


def get_issuer(document_path: str) -> str:
    with open(document_path, "r") as f:
        loaded_json = json.loads(f.read())

    loaded_json_key_value_pairs = loaded_json["key-value_pairs"]  # Issuer: null
    try:
        issuer = loaded_json_key_value_pairs["Issuer"][0]
    except KeyError as e:
        issuer = None
        # print(f'KeyError: {e}')

    documents = loaded_json["full_text"]
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0.2, separator=" "
    )
    texts = text_splitter.split_text(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(
        "Who is the issuer of this term sheet?",
    )
    prompt = PromptTemplate.from_template(
        "Take a deep breath and relax. Think step by step."
        "I have the following  document of a term sheet"
        "I need to find the Issuer (which is basically the company issuing the term sheet)"
        "The format of the issuer is usually an acronym of the company name"
        "It is really important that you get this right, because my life depends on it!\n"
        "Example:"
        "Context: Issuer: BNP Paribas Issuance B.V. (S&P's A+)"
        "Issuer: BNP\n\n"
        "I am going to give you a chunk of text, and you need to tell me which one is the issuer of the document\n\n"
        "Context:{context}\n"
        "Issuer: <your answer here>"
    )
    full_context = ""
    for doc in docs:
        full_context += "\n" + doc.page_content
    if issuer:
        try:
            issuer_confidence = loaded_json_key_value_pairs["Issuer"][1]
        except KeyError as e:
            issuer_confidence = None
            # print(f'No Issuer found in key value pairs: {e}')
        full_context += (
            "\n"
            + f" I also did an OCR anaysis and found Issuer: {issuer} with a confidence of {issuer_confidence}"
        )

    final_prompt = prompt.format(context=full_context)
    result = instruction_response(final_prompt)  # issuer: the issuer is bnp
    # if regex issuer in result
    if "Issuer:" in result:
        temp = result.split(":")[1].replace("\n", "")
        final_result = temp.replace("\n", "")
    else:
        final_result = result.replace("\n", "")

    issuer_llm = final_result
    pattern = r"\b[A-Z][A-Za-z]*\b"
    # Find all matches
    matches = re.findall(pattern, issuer_llm)
    final_check = PromptTemplate.from_template(
        "I am about to provide the final issuer, remember that my life depends on it!\n"
        "The repsonse needs to be an acronym of the company name. "
        "Example: "
        "Issuer: ['BNP Paribas Issuance B.V. (S&P's A+)'"
        "BNP"
        " Now its time for you to respond."
        "Remember the response should be a single string of the acronym of the company name\n\n"
        "Issuer: {final_issuer}"
    )
    final_check_prompt = final_check.format(final_issuer=matches)
    final_check_result = instruction_response(final_check_prompt)
    if ":" in final_check_result:
        temp = final_check_result.split(":")[1].replace("\n", "")
        final_check_result = temp.replace("\n", "")
    final_result = final_check_result.replace("\n", "")
    return final_result


# puntos de mejora:
# hacer un aanalysis entero de key value pairs
# se puede comprobar el calculation agent
# se puede comprobar el issuer en los key value pairs
