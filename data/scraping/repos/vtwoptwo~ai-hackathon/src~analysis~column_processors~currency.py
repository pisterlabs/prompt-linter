from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import re
from .helpers.utils import instruction_response, retry_on_rate_limit_error

# list of all currencies:world_currencies
currencies = [
    "AED",
    "AFN",
    "ALL",
    "AMD",
    "ANG",
    "AOA",
    "ARS",
    "AUD",
    "AWG",
    "AZN",
    "BAM",
    "BBD",
    "BDT",
    "BGN",
    "BHD",
    "BIF",
    "BMD",
    "BND",
    "BOB",
    "BRL",
    "BSD",
    "BTN",
    "BWP",
    "BYN",
    "BZD",
    "CAD",
    "CDF",
    "CHF",
    "CLP",
    "CNY",
    "COP",
    "CRC",
    "CUC",
    "CUP",
    "CVE",
    "CZK",
    "DJF",
    "DKK",
    "DOP",
    "DZD",
    "EGP",
    "ERN",
    "ETB",
    "EUR",
    "FJD",
    "FKP",
    "FOK",
    "GBP",
    "GEL",
    "GGP",
    "GHS",
    "GIP",
    "GMD",
    "GNF",
    "GTQ",
    "GYD",
    "HKD",
    "HNL",
    "HRK",
    "HTG",
    "HUF",
    "IDR",
    "ILS",
    "IMP",
    "INR",
    "IQD",
    "IRR",
    "ISK",
    "JEP",
    "JMD",
    "JOD",
    "JPY",
    "KES",
    "KGS",
    "KHR",
    "KID",
    "KMF",
    "KPW",
    "KRW",
    "KWD",
    "KYD",
    "KZT",
    "LAK",
    "LBP",
    "LKR",
    "LRD",
    "LSL",
    "LYD",
    "MAD",
    "MDL",
    "MGA",
    "MKD",
    "MMK",
    "MNT",
    "MOP",
    "MRU",
    "MUR",
    "MVR",
    "MWK",
    "MXN",
    "MYR",
    "MZN",
    "NAD",
    "NGN",
    "NIO",
    "NOK",
    "NPR",
    "NZD",
    "OMR",
    "PAB",
    "PEN",
    "PGK",
    "PHP",
    "PKR",
    "PLN",
    "PYG",
    "QAR",
    "RON",
    "RSD",
    "RUB",
    "RWF",
    "SAR",
    "SBD",
    "SCR",
    "SDG",
    "SEK",
    "SGD",
    "SHP",
    "SLL",
    "SOS",
    "SRD",
    "SSP",
    "STN",
    "SVC",
    "SYP",
    "SZL",
    "THB",
    "TJS",
    "TMT",
    "TND",
    "TOP",
    "TRY",
    "TTD",
    "TVD",
    "TWD",
    "TZS",
    "UAH",
    "UGX",
    "USD",
    "UYU",
    "UZS",
    "VES",
    "VND",
    "VUV",
    "WST",
    "XAF",
    "XCD",
    "XDR",
    "XOF",
    "XPF",
    "YER",
    "ZAR",
    "ZMW",
    "ZWL",
]

load_dotenv()


@retry_on_rate_limit_error(wait_time=10)
def get_currency(document_path: str) -> str:
    # read the document
    with open(document_path, "r") as f:
        loaded_json = json.load(f)

    # Creating the regex pattern to check for all currencies
    pattern = r"\b(?:" + "|".join(currencies) + r")\b"
    matches = re.findall(pattern, loaded_json["full_text"])
    # get unique matches
    matches = list(set(matches))
    if len(matches) == 1:
        # edge case that only one currency is mentioned
        return matches[0]
    else:
        documents = loaded_json["full_text"]
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, separator=" "
        )
        texts = text_splitter.split_text(documents)
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_texts(texts, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 10})
        docs = retriever.get_relevant_documents("Currency")
        template = PromptTemplate.from_template(
            "I am going to give you the text of a term sheet. "
            "Give me the resulting currency in which the money is traded of this term sheet\n\n"
            "Context:{context}\n"
            "Take into account that the data from OCR and regex can be very exact"
            "Your Response:"
            "<your answer here>"
        )

        full_context = ""
        for doc in docs:
            full_context += "\n" + doc.page_content
        if matches:
            full_context += (
                "\n"
                + f" I also did search for currencies and found the following matches; {matches}"
            )
        if "Currency" in loaded_json["key-value_pairs"].keys():
            currency = loaded_json["key-value_pairs"]["Currency"][0]
            currency_confidence = loaded_json["key-value_pairs"]["Currency"][1]
            full_context += (
                "\n"
                + f" I also did an OCR anaysis and found Currency: {currency} with a confidence of {currency_confidence}"
            )

        context = template.format(context=full_context)
        result = instruction_response(context)
        currency = result
        final_response = re.findall(pattern, currency)

        final_check = PromptTemplate.from_template(
            "I am about to provide you with the final response of the currency used in a term sheet."
            "The final response should be of the abbreviation of the currency"
            "The final response is {currency}"
            "What is the response in the desired format (abbreviation of the currency)?"
            "If not respond with the abbreviation of the currency"
            "<your answer here>"
        )

        if final_response:
            final_response = final_response[0]
            final_response = final_response.upper()
            final_response = final_check.format(currency=final_response)
            result = instruction_response(final_response)
            final_response = result
            final_response = re.findall(pattern, final_response)
            final_response = final_response[0]
            final_response = final_response.upper()

        return final_response
