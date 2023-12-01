# This is the first simple example from the blog post that processes data
# from Wikipedia and does not use orchestration
import os
import sys
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
import pathlib
import subprocess
import tempfile

def get_github_docs(repo_owner, repo_name):
    with tempfile.TemporaryDirectory() as d:
        subprocess.check_call(
            f"git clone --depth 1 https://yessicavrm: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@github.com/{repo_owner}/{repo_name}.git .",
            cwd=d,
            shell=True,
        )
        git_sha = (
            subprocess.check_output("git rev-parse HEAD", shell=True, cwd=d)
            .decode("utf-8")
            .strip()
        )
        repo_path = pathlib.Path(d)
 
        
        markdown_files = list(repo_path.glob("**/*.md")) + list(repo_path.glob("**/*.mdx"))

        print("-----------------")
        print(markdown_files)
        print("-----------------")

        for markdown_file in markdown_files:
            with open(markdown_file, "r") as f:
                relative_path = markdown_file.relative_to(repo_path)
                github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/{git_sha}/{relative_path}"
                yield Document(page_content=f.read(), metadata={"source": github_url})


sources = get_github_docs("h2oai", "hydrogen_torch")


source_chunks = []
splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)


counter = 1
for source in sources:
    counter = counter + 1
    print(counter)
    for chunk in splitter.split_text(source.page_content):
        source_chunks.append(Document(page_content=chunk, metadata=source.metadata))


search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())


chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="map_reduce")


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
