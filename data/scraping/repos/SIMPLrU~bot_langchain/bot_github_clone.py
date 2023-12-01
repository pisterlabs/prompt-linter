# This is the first simple example from the blog post that processes data
# from Wikipedia and does not use orchestration

import requests
import pathlib
import subprocess
import tempfile

from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter

## Checkout Github repo and crawl for markdown files, return LangChang documents
def get_github_docs(repo_owner, repo_name):
    with tempfile.TemporaryDirectory() as d:
        subprocess.check_call(
            f"git clone --depth 1 https://github.com/{repo_owner}/{repo_name}.git .",
            cwd=d,
            shell=True,
        )
        git_sha = (
            subprocess.check_output("git rev-parse HEAD", shell=True, cwd=d)
            .decode("utf-8")
            .strip()
        )
        repo_path = pathlib.Path(d)
        markdown_files = list(repo_path.glob("*/*.md")) + list(
            repo_path.glob("*/*.mdx")
        )
        for markdown_file in markdown_files:
            with open(markdown_file, "r") as f:
                relative_path = markdown_file.relative_to(repo_path)
                github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/{git_sha}/{relative_path}"
                yield Document(page_content=f.read(), metadata={"source": github_url})

### Hook Repo to Bot
sources = get_github_docs("dagster-io", "dagster")
# sources = get_github_docs("jerryjliu", "gpt_index")
# sources = get_github_docs("Kong", "docs.konghq.com")


##########################
##### Crawl Wiki URL #####
###########################
## Next, we’ll need some sample data for our toy example. For now, let’s use the first paragraph of various Wikipedia pages as our data sources. There’s a great Stack Overflow answer that gives us a magic incantation to fetch this data:
# def get_wiki_data(title, first_paragraph_only):
#     url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles={title}"
#     if first_paragraph_only:
#         url += "&exintro=1"
#     data = requests.get(url).json()
#     return Document(
#         page_content=list(data["query"]["pages"].values())[0]["extract"],
#         metadata={"source": f"https://en.wikipedia.org/wiki/{title}"},
#     )

# Define list of wikipedia articles as sources
# sources = [
#     get_wiki_data("Unix", True),
#     get_wiki_data("Microsoft_Windows", True),
#     get_wiki_data("Linux", True),
#     get_wiki_data("Seinfeld", True),
#     get_wiki_data("Matchbox_Twenty", True),
#     get_wiki_data("Roman_Empire", True),
#     get_wiki_data("London", True),
#     get_wiki_data("Python_(programming_language)", True),
#     get_wiki_data("Monty_Python", True),
#     get_wiki_data("Microservices", True),
# ]
##########################
##### Crawl Wiki URL #####
###########################

# Initialize list to hold chunks of text from sources
source_chunks = []

# Initialize text splitter to divide sources into chunks
splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)

# Iterate over sources and split into chunks, appending each chunk to source_chunks
for source in sources:
    for chunk in splitter.split_text(source.page_content):
        source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

# Create search index using FAISS and OpenAIEmbeddings
search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())

# load_qa_with_sources_chain from Langchain
chain = load_qa_with_sources_chain(OpenAI(temperature=0))

# Define function to print answer
def print_answer(question):
    print(
        chain(
            {
                "input_documents": search_index.similarity_search(question, k=4),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    )
