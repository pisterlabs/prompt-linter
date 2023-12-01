from openai.embeddings_utils import cosine_similarity
from openai.embeddings_utils import get_embedding
import sys
import os
from glob import glob
import pandas as pd
import openai

from tree_sitter import Language, Parser

RESULTS_CHAT_WITH = 200
MAX_TOKENS = 6500  # 8191
MAX_COMBINED_RESULTS_LENGTH = 6000
CACHED_EMBEDDINGS_PATH = "code_search_openai.json"


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


GO_LANGUAGE = (Language('build/my-languages.so', 'go'), "*.go")
JS_LANGUAGE = (Language('build/my-languages.so', 'javascript'), "*.go")
PY_LANGUAGE = (Language('build/my-languages.so', 'python'), "*.py")

current_language = GO_LANGUAGE

openai.api_key = os.environ['OPENAI_API_KEY']

# path to code directory to search/embed/query
code_root = None
if os.path.exists(CACHED_EMBEDDINGS_PATH):
    with open("previous_search_path.txt", "r") as file:
        string_variable = file.read()
else:
    print("Please type in the path to the code directory you want to search/embed/query:")
    code_root = input()
    if code_root.strip() == "":
        code_root = os.getcwd()
    with open("previous_search_path.txt", "w") as file:
        file.write(code_root)

print("Type in your query/prompt:")
code_query = input()


def get_functions(filepath):
    """
    Get all functions in a Python file.
    """
    codestr = open(filepath).read().replace("\r", "\n")

    parser = Parser()
    parser.set_language(current_language[0])

    tree = parser.parse(bytes(codestr, "utf8"))

    cursor = tree.walk()

    cursor.goto_first_child()

    while True:
        print("type: ", cursor.node.type)
        print("byte locations: ", cursor.node.start_byte,
              " - ", cursor.node.end_byte)
        code = codestr[cursor.node.start_byte:cursor.node.end_byte]
        node_type = cursor.node.type
        print("code:\n", code)
        code_filename = {
            "code": code, "node_type": node_type, "filepath": filepath}
        if code.strip() != "":
            print("code_filename: ", code_filename)
            yield code_filename
        has_sibling = cursor.goto_next_sibling()
        if not has_sibling:
            break

df = None
if os.path.exists(CACHED_EMBEDDINGS_PATH):
    print("\n\n\nWARNING: USING CACHED EMBEDDINGS!!!\n\n\n")
    df = pd.read_json(CACHED_EMBEDDINGS_PATH)
else:
    code_files = [y for x in os.walk(code_root)
                for y in glob(os.path.join(x[0], current_language[1]))]

    print("Total number of files found:", len(code_files))

    if len(code_files) == 0:
        print("Double check that you have downloaded the openai-python repo and set the code_root variable correctly.")

    all_nodes = []
    for code_file in code_files:
        nodes = list(get_functions(code_file))
        for func in nodes:
            all_nodes.append(func)

    node_count = len(all_nodes)
    print("Total number of functions extracted:", node_count)

    df = pd.DataFrame(all_nodes)
    df['code_embedding'] = df['code'].apply(
        lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    print("Done creating embeddings")
    df['filepath'] = df['filepath'].apply(lambda x: x.replace(code_root, ""))
    df.to_json(CACHED_EMBEDDINGS_PATH)

# print("\nEMBEDDINGS ARE:\n", df['code_embedding'])
print("Generated or loaded all embeddings")


query_embedding = get_embedding(code_query, engine='text-embedding-ada-002')
if not query_embedding:
    print("Error: query embedding is empty")
    sys.exit(1)


def search_functions(df, number_similar_results=200, n_lines=1000):
    df['similarities'] = df.code_embedding.apply(
        lambda x: cosine_similarity(x, query_embedding))

    res = df.sort_values('similarities', ascending=False).head(
        number_similar_results)

    combined_results = ""
    srs = []
    for r in res.iterrows():
        code = "\n".join(r[1].code.split("\n")[:n_lines])
        srs.append([r[1].filepath+":"+r[1].node_type,
                   r[1].similarities, code])
        print(r[1].filepath+":"+r[1].node_type +
              "  score=" + str(round(r[1].similarities, 3)))
        print(code)
        print('-'*70)
        combined_results += code + "\n\n\n"
    srs.sort(key=lambda x: x[1], reverse=True)
    print("\n\n\n\n**************************************************************\n")
    print("Best ranking code snippet:")
    print(srs[0][2])
    # return res
    return combined_results[:MAX_COMBINED_RESULTS_LENGTH]


print("Running search functions to find similar code")
related_code = search_functions(df, number_similar_results=RESULTS_CHAT_WITH)
header = """Answer the question using the provided context and any other available information."\n\nContext:\n"""
final_prompt = header + \
    "".join(related_code) + "\n\n Q: " + code_query + "\n A:"

final_answer = openai.ChatCompletion.create(
    messages=[{"role": "user", "content": final_prompt}],
    model="gpt-3.5-turbo"
)

print('-'*70)
print('-'*70)
print("\n\n\nChatGPT says:\n\n", final_answer.choices[0].message.content)
