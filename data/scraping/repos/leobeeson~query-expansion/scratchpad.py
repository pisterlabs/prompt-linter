import os
import random
import time
import tiktoken
import pandas as pd


from dotenv import load_dotenv, find_dotenv


from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI


from src.dal.category_metadata import CategoryMetadata
from src.dal.query_set import QuerySet


_ = load_dotenv(find_dotenv())


# encoding = tiktoken.get_encoding("cl100k_base")
# encoding_3_5 = tiktoken.encoding_for_model('gpt-3.5-turbo')
# encoding_4 = tiktoken.encoding_for_model('gpt-4')


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


random.seed(235)
def sample_elements(my_list, num_samples):
    return random.sample(my_list, min(num_samples, len(my_list)))


data = pd.read_csv("./data/trainSet.csv", header=None, names=["query", "category"])

category_id = 24

search_queries = data[data["category"] == category_id]["query"].tolist()
search_queries.sort()
len(search_queries) #534
search_queries_sample = sample_elements(search_queries, 400)
len(search_queries_sample) #400
search_queries_paragraph = "; ".join(search_queries_sample)
num_tokens_from_string(search_queries_paragraph, "cl100k_base") #1,382
# len(encoding_3_5.encode(search_queries_paragraph))
# len(encoding_4.encode(search_queries_paragraph))


parser = PydanticOutputParser(pydantic_object=CategoryMetadata)
parser_instructions = parser.get_format_instructions()
num_tokens_from_string(parser_instructions, "cl100k_base") #565

instructions = """
You are a market analyst working for a company that provides market intelligence for search engine optimization.
Your task is to:
1) First understand a conceptual model for the market category encompassed by the search queries below surrounded in triple backticks
2) Second, generate a taxonomy that best clusters the category into its most relevant subcategories.
3) Third, for each subcategory, generate a label, description, list of subtopics, list of noun phrases, and list of named entities representing products, services, materials, brands, and companies. Keep all labels, noun phrases, entities, and subtopics to ideally three words long, and a maximum of five words Long.
SEARCH QUERIES:
```{search_queries}```
Please structure the taxonomy and its components in the following format below surrounded in triple pound signs:
RESPONSE FORMAT:
###{format_instructions}###
"""

# Follow the instructions below step by step to generate a response in the above output format:
# 1. Compare each search query to each other and understand the market category they represent.
# 2. Generate a list of subcategories, estimate their relevance to the market category with a value between 0 and 1, sort them from most to least relevant, and select maximum the ten most relevant.
# 3. For each subcategory, generate a label, description, list of topics, list of nouns, and list of entities representing products, services, materials, brands, and companies.
# 4. For every one of the above lists, estimate the relevance of each item to the subcategory with a value between 0 and 1, sort them from most to least relevant, and select maximum the ten most relevant.

# num_tokens_from_string(instructions, "cl100k_base") #69

prompt = PromptTemplate(
    template=instructions,
    input_variables=["search_queries"],
    partial_variables={"format_instructions": parser_instructions},
    output_parser=parser
)

_input = prompt.format_prompt(search_queries=search_queries_paragraph)
_input.to_string()
num_tokens_from_string(_input.to_string(), "cl100k_base") #2719 -> cat 24


chat = ChatOpenAI(temperature=0)
llm_chain = LLMChain(
    llm=chat,
    prompt=prompt
)

t0 = time.time()
response = llm_chain.predict_and_parse(search_queries=search_queries_paragraph, return_only_outputs=True)
t1 = time.time()
print(t1 - t0)

# cat 24 (n=400): 52 secs
# cat 24 (n=461): 35.6 secs
# cat 587 (n=4): 35 secs
type(response)
response.dict()
response.json()


def main():
    pass    


if __name__ == "__main__":
    main()
