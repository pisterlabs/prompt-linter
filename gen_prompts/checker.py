import glob
import json

from tree_sitter import Language, Parser, Tree

PY_LANGUAGE = Language("./build/my-languages.so", "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)


def print_imports(filename):
    with open(filename, "rb") as f:
        tree = parser.parse(f.read())
    query = PY_LANGUAGE.query("(import_statement) @import")
    query2 = PY_LANGUAGE.query("(import_from_statement) @import")

    for import_ in query.captures(tree.root_node):
        print(import_[0].text.decode("utf-8"))

    for import_ in query2.captures(tree.root_node):
        print(import_[0].text.decode("utf-8"))


with open("openai_repos.txt") as f:
    lines = f.readlines()

with open("openai-calls.json") as f:
    calls = list(map(lambda x: x[0], json.load(f)))

with open("chat-calls.json") as f:
    calls.extend(list(map(lambda x: x[0], json.load(f))))

with open("langchain-calls.json") as f:
    calls.extend(list(map(lambda x: x[0], json.load(f))))

with open("tool-strings.json") as f:
    calls.extend(list(map(lambda x: x[0], json.load(f))))

with open("prompt_or_template_in_name-strings.json") as f:
    calls.extend(list(map(lambda x: x[0], json.load(f))))

print("Documents which matched at least one pattern:", len(calls))

# for folder in glob.glob("data/scraping/repos/*"):
#     for file in glob.glob(folder + "/*.py"):
#         if file not in calls:
#             print_imports(file)
#             print(file)
#             breakpoint()

# for line in lines:
#     if line.strip() not in calls:
#         print(line.strip())
