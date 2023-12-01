import json
import os
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

# Load and run summarization chain
# https://python.langchain.com/en/latest/modules/chains/index_examples/summarize.html
# The recommended way to get started using a summarization chain is: map_reduce
# different chain types: stuff, map_reduce, and refine
# https://docs.langchain.com/docs/components/chains/index_related_chains
from prompt_template import chain_map_reduce_template, chain_refine_template


# Load the transcript & language from the JSON file
def get_json_content():
    with open('sample.json') as file:
        return json.load(file)


def make_summary(input_docs, language):
    # Initialize OpenAI model and text splitter
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # chain = load_summarize_chain(llm, chain_type="refine", map_prompt=prompt, combine_prompt=prompt)
    # verbose=True testing

    prompt_template = chain_map_reduce_template()
    chain = load_summarize_chain(llm, verbose=True, return_intermediate_steps=True, **prompt_template)
    values = chain({"input_documents": input_docs, "language": language}, return_only_outputs=True)
    return values


# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

content = get_json_content()
source = content['results']['transcripts'][0]['transcript']
# extracts the language_code
language_code = content['results']['language_code']
source_count = len(source.split())
print(f"Source count:{source_count}")

text_splitter = CharacterTextSplitter(separator="\n")

# Split text into smaller chunks and create Document objects
texts = text_splitter.split_text(source)

docs = [Document(page_content=t) for t in texts]

print(f"Document count:{len(docs)}")

summary = make_summary(docs, "English")

print(summary)
# #
# summary_count = len(summary.split())
# print(f"Summary count:{summary_count}")
