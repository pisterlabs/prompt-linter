import pprint
import time
import os
from langchain.chat_models import ChatOpenAI
from langchain.document_transformers import (
    beautiful_soup_transformer,
)
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import AsyncChromiumLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain
from dotenv import load_dotenv

load_dotenv()

schema = {
    "properties": {
        "person_name": {"type": "string"},
        "service_provided": {"type": "string"},
        "service_location": {"type": "string"},
        "service_price": {"type": "string"},
        "contact email": {"type": "string"},
        "contact number": {"type": "string"},
    },
    "required": ["person_name", "news_article_summary"],
}

embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), chunk_size=1000)


def scrape_with_playwright(urls):
    t_flag1 = time.time()
    loader = AsyncChromiumLoader(urls)

    docs = loader.load()
    t_flag2 = time.time()
    print("AsyncChromiumLoader time: ", t_flag2 - t_flag1)

    bs_transformer = beautiful_soup_transformer.BeautifulSoupTransformer()

    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["p", "li", "div", "a", "span"]
    )
    t_flag3 = time.time()
    print("BeautifulSoupTransformer time: ", t_flag3 - t_flag2)
    print("Extracting content with LLM")

    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800, chunk_overlap=50
    )
    t_flag4 = time.time()
    print("RecursiveCharacterTextSplitter time: ", t_flag4 - t_flag3)
    splits = splitter.split_documents(docs_transformed)

    print(len(splits))
    return splits


def embedder(texts):
    start_embedding = time.time()
    emb_result = embeddings.embed_documents(texts)
    end_embedding = time.time()
    print("Embedding time: ", end_embedding - start_embedding)
    return emb_result


input = """Give list of bus services in Kerala for wedding family trip, with correct name and contact details"""


def extract(schema, content):
    MY_ENV_VAR = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        temperature=0, model="gpt-3.5-turbo", api_key=MY_ENV_VAR, max_tokens=1000, openai_api_key=MY_ENV_VAR, streaming=True
    )
    chain = create_extraction_chain(schema, llm)
    chain.run(content)
    return chain

prompt = "I want to get a bus service for wedding family trip in Kerala"

urls = [
    "https://keralatourbus.com/",
    "https://www.justdial.com/Thrissur/Bus-On-Rent-For-Wedding-in-Kerala/nct-11275471",
    "https://www.sulekha.com/bus-rentals/trivandrum",
    "https://www.shaadibaraati.com/wedding-transportation/kerala/oneness-travels/MGyvjmIxtV",
    "https://devannmpd.wixsite.com/taxicarkerala/tourist-buses-in-cochin",
    "https://www.asparkholidays.com/cochin/luxury-bus-hire",
    "https://www.redbus.in/bus-hire/wedding",
    "https://www.asparkholidays.com/thiruvananthapuram/book-volvo-bus",
    "https://www.justdial.com/Ernakulam/Bus-On-Rent-For-Wedding/nct-11275471",

    "https://scikit-learn.org/stable/install.html",
    "https://www.nvidia.com/en-in/data-center/a100/",
    "https://www.nature.com/articles/s41467-023-43491-w",
    "https://www.theregister.com/2023/11/23/medley_interlisp_revival/",
    "https://www.science.org/doi/10.1126/science.adm9964",
    "https://www.techradar.com/pro/the-gpt-to-rule-them-all-training-for-one-trillion-parameter-model-backed-by-intel-and-us-government-has-just-begun",
    "https://designsprintkit.withgoogle.com/methodology/phase3-sketch/crazy-8s"

]
extracted_content = scrape_with_playwright(urls)

# write the extrated data in a file
with open("split.txt", "w") as f:
    f.write(str(extracted_content))


data = [chunk.page_content for chunk in extracted_content]

# embedd data
embedded_data = embedder(data)
db = Chroma.from_documents(extracted_content, embeddings)


start_search = time.time()
docs = db.similarity_search(input, k=1)
end_search = time.time()
print("Search time: ", end_search - start_search)

print(docs[0].page_content)

gen_data = extract(schema, docs[0].page_content)

print(gen_data)