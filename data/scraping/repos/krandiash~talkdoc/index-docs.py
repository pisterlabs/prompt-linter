import argparse
import os
import pathlib
from functools import partial
import subprocess
from typing import Callable, List

import cohere
import git
import meerkat as mk
import openai
import tiktoken
from rich import print

ENCODING = tiktoken.get_encoding("gpt2")

openai.api_key = os.environ.get("OPENAI_API_KEY")
if os.environ.get("COHERE_API_KEY"):
    co = cohere.Client(os.environ.get("COHERE_API_KEY"))


def get_files_in_repo(repo_path: str) -> list:
    """
    Given a path to a Git repository on the local filesystem,
    return a list of files in the repository.

    Only files that are tracked by Git are returned.
    """
    # Create a Git repository object
    print(f"Loading Git repository at {repo_path}")
    repo = git.Repo(repo_path)

    # Get the list of files of interest
    files = repo.git.ls_files(
        "--exclude-standard", "--cached", "--modified", "--other"
    ).splitlines()
    paths = [os.path.join(repo_path, f) for f in files]

    # For each file, get {'filename', 'len', 'extension'}
    files = [
        {
            "filename": f,
            "nchars": os.path.getsize(f),
            "extension": pathlib.Path(f).suffix,
        }
        for f in paths
    ]

    return files


def get_token_count(files):
    """
    Given a list of files, return the number of tokens in each file.
    """
    return [len(e) for e in ENCODING.encode_batch(files)]


def construct_project_dataframe(files) -> mk.DataFrame:
    """
    Given a list of files, construct a Meerkat DataFrame.
    """
    project = mk.DataFrame(files)
    project.create_primary_key("file_id")

    # Add a column that contains the actual file
    project["files"] = mk.files(project["filename"], type="code")

    # Go through all the files in order, keep track of failed loads and their extensions
    failed_extensions = set()
    for i in range(len(project)):
        try:
            project["files"][i]()
        except Exception:
            failed_extensions.add(project["extension"][i])

    # Make a list of image and pdf file extensions
    remove_extensions = [
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".pdf",
        ".ico",
        "",
        ".dia",
        ".odg",
        ".pkl",
        ".npz",
        ".fits",
        ".mod",
        ".swg",
        ".star",
        ".npy",
    ] + list(failed_extensions)
    # remove_extensions = set(remove_extensions) # this doesn't work?!

    # Exclude files that are images or pdfs
    project = project.filter(
        lambda extension: extension not in remove_extensions, pbar=True
    )

    # Exclude files that are empty
    project = project.filter(
        lambda row: row["nchars"] > 0, materialize=False, pbar=True
    )

    # Add a column that contains the number of tokens in each file
    project["ntokens"] = project["files"].map(
        get_token_count, batch_size=128, is_batched_fn=True, pbar=True
    )
    print("Total Tokens: {}".format(sum(project["ntokens"])))

    return project


def chunker(files: List[str], toksize: int = 2048) -> List[str]:
    """Split each file into chunks of size toksize."""
    # Get the encoding
    encoding = tiktoken.get_encoding("gpt2")
    # Tokenized files
    tokens = encoding.encode_batch(files)
    # Split each file into chunks of size toksize
    splits = [
        [encoding.decode(e[pos : pos + toksize]) for pos in range(0, len(e), toksize)]
        for e in tokens
    ]
    return splits


def explode(
    df: mk.DataFrame,
    chunk_col: str,
    chunker: Callable,
    batch_size: int = 1,
) -> mk.DataFrame:
    """Chunk each row of a DataFrame into multiple rows, and concatenate the results."""
    # Chunk each row of the DataFrame
    chunks = df.map(
        chunker,
        batch_size=batch_size,
        is_batched_fn=batch_size > 1,
        pbar=True,
        inputs={chunk_col: "files"},
    )
    df["chunks"] = chunks

    # Make a df on each row, propagate the other columns
    chunk_dfs = df.map(
        lambda row: mk.DataFrame(
            {
                "chunk": row["chunks"],
                "chunk_idx": list(range(1, len(row["chunks"]) + 1)),
                **{
                    k: [v] * len(row["chunks"])
                    for k, v in row.items()
                    if k in ["filename", "file_id"]
                },
            }
        ),
        pbar=True,
    )

    # Concatenate the results
    return mk.concat(chunk_dfs)


def prepare_chunk_dataframe(df: mk.DataFrame) -> mk.DataFrame:
    """ """
    # Chunk each file into 2048-token chunks
    chunk_df = explode(df, "files", partial(chunker, toksize=2048), batch_size=16)
    chunk_df.create_primary_key("chunk_id")

    # Add a formatter, this helps with visualization in notebooks
    chunk_df["chunk"] = chunk_df["chunk"].format(mk.format.CodeFormatterGroup())

    # Add token counts
    chunk_df["ntokens"] = chunk_df["chunk"].map(
        get_token_count, batch_size=16, is_batched_fn=True, pbar=True
    )

    return chunk_df


def embed(text, model="openai/text-embedding-ada-002"):
    text = text.replace("\n", " ")
    if model.startswith("openai"):
        response = openai.Embedding.create(
            input=[text], model=model.replace("openai/", "")
        )
        return response["data"][0]["embedding"]
    elif model.startswith("cohere"):
        response = co.embed(texts=[text], model=model.replace("cohere/", ""))
        return response.embeddings[0]


def embed_many(texts, model="openai/text-embedding-ada-002"):
    texts = [t.replace("\n", " ") for t in texts]
    if model.startswith("openai"):
        response = openai.Embedding.create(
            input=texts, model=model.replace("openai/", "")
        )
        return [response["data"][i]["embedding"] for i in range(len(texts))]
    elif model.startswith("cohere"):
        response = co.embed(texts=texts, model=model.replace("cohere/", ""))
        return response.embeddings


def search(
    df,
    query,
    n,
    embedding_col: str,
    model: str = "openai/text-embedding-ada-002",
):
    # Embed the query
    query_embedding = embed(query, model=model)
    # Compute the cosine similarity between the query and each chunk
    similarities = df[embedding_col].dot(query_embedding)
    # Sort the chunks by similarity
    df["similarity"] = similarities
    df = df.sort("similarity", ascending=False)
    # Return the top n results
    return df.head(n)


def template(instruction, query, context):
    return f"""
{instruction}

Query: {query}

Relevant Context:
{context}

Helpful Response:\
"""


def truncate(text: str, ntokens: int):
    """Truncate a string to a number of tokens."""
    tokens = ENCODING.encode(text)[:ntokens]
    print(f"Truncated to {len(tokens)} tokens")
    return ENCODING.decode(tokens)


def create_prompt(
    df,
    query,
    embedding_col,
    n=2,
    max_tokens=6144,
    model="openai/text-embedding-ada-002",
):
    # Search for the query
    results = search(df, query, n=n, embedding_col=embedding_col, model=model)
    # Create the prompt
    instruction = "Please provide a helpful response to the following query"
    " using the provided context. Your response should be well formatted, "
    "and can include Python code snippets, but must use the library."
    context = truncate("\n\n".join(results["chunk"]), max_tokens)
    return template(instruction, query, context)


def main(repo_path: str, model: str = "openai/text-embedding-ada-002"):
    # Get a list of files in the repository
    files: List[str] = get_files_in_repo(repo_path)

    # Ask the user to confirm whether to continue
    print(f"Found {len(files)} files in {repo_path}")
    if not input("Continue? [y/n] ").lower().startswith("y"):
        exit()

    # Construct a Meerkat DataFrame
    project: mk.DataFrame = construct_project_dataframe(files)

    # Chunk each file into 2048-token chunks
    chunks: mk.DataFrame = prepare_chunk_dataframe(project)

    # Ask the user to confirm whether to continue
    print(f"Embedding all {len(chunks)} chunks")
    if not input("Continue? [y/n] ").lower().startswith("y"):
        exit()

    # Embed each chunk for retrieval
    chunks[f"embeddings/{model}"] = chunks.map(
        lambda chunk: embed_many(chunk, model=model),
        pbar=True,
        batch_size=128,
        is_batched_fn=True,
        output_type=mk.TensorColumn,
    )

    return project, chunks


def interface(
    repo_name: str,
    chunks: mk.DataFrame,
    model: str = "openai/text-embedding-ada-002",
    prompt_only: bool = False,
):
    # Run an interactive prompt with the user.
    # The user can enter a query, and the interface will display the prompt, and then the result of
    # asking ChatGPT to complete the prompt.
    while True:
        print("=====================")
        query = input("Enter a query: ")
        n = int(input("Enter the number of chunks to retrieve: "))
        if not query:
            continue
        print("=====================")
        prompt = create_prompt(
            chunks,
            query,
            embedding_col=f"embeddings/{model}",
            n=n,
            model=model,
            max_tokens=3072,
        )
        if prompt_only:
            print(prompt)
            continue
        else:
            print("=====================")
            print("Response from ChatGPT")
            print("=====================")
        for response in openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that helps"
                    f" developers understand and use the {repo_name} code"
                    " documentation to complete their tasks. Make sure you"
                    f" explain {repo_name} code to the user (and don't confuse"
                    " them with information about other libraries).",
                },
                {"role": "user", "content": prompt},
            ],
            stream=True,
        ):
            try:
                print(response["choices"][0]["delta"]["content"], end="", flush=True)
            except KeyError:
                pass
        print()
        print("=====================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str)
    parser.add_argument("--model", type=str, default="openai/text-embedding-ada-002")
    parser.add_argument("--prompt-only", action="store_true")

    args = parser.parse_args()

    # If the repo is a URL, clone it
    repo_name = os.path.basename(args.repo).replace(".git", "")
    print(f"Using repo {repo_name}")

    if os.path.exists(f"./{repo_name}-project.mk"):
        project = mk.DataFrame.read(f"./{repo_name}-project.mk")
        chunks = mk.DataFrame.read(f"./{repo_name}-chunks.mk")    
    else:
        if args.repo.startswith("http"):
            if os.path.exists(repo_name):
                print(f"Repo {repo_name} already exists, skipping clone")
            else:
                print(f"Cloning repo {repo_name}")
                subprocess.run(["git", "clone", args.repo])

            # Point the repo to the local path
            args.repo = os.getcwd() + "/" + repo_name

        project, chunks = main(args.repo, model=args.model)
        project.write(f"./{repo_name}-project.mk")
        chunks.write(f"./{repo_name}-chunks.mk")

    interface(repo_name, chunks=chunks, model=args.model, prompt_only=args.prompt_only)
