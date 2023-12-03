import os
import json

from tree_sitter import Language, Parser, Tree

PY_LANGUAGE = Language("./build/my-languages.so", "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)


def parse_tree(tree: Tree):
    strings = []
    identifiers = []
    interpolations = []

    query = PY_LANGUAGE.query("(string) @string")
    for string in query.captures(tree.root_node):
        if string[0].text.decode("utf-8").count(" ") == 0:
            continue

        strings.append(string[0].text.decode("utf-8"))

    query = PY_LANGUAGE.query("(identifier) @identifier")
    for identifier in query.captures(tree.root_node):
        identifiers.append(identifier[0].text.decode("utf-8"))

    query = PY_LANGUAGE.query("(interpolation) @identifier")
    for interpolation in query.captures(tree.root_node):
        interpolations.append(interpolation[0].text.decode("utf-8"))

    return strings, identifiers, interpolations


def parse_prompts(filename):
    with open(filename) as f:
        prompts = f.read().split("\n----------\n")

    prompts = list(map(lambda x: x.strip(), prompts))
    prompts = list(filter(lambda x: x != "", prompts))

    print("Found", len(prompts), "prompts")

    data = {}
    for prompt in prompts:
        tree = parser.parse(bytes(prompt, "utf-8"))
        strings, identifiers, interpolations = parse_tree(tree)
        data[prompt] = {
            "strings": strings,
            "identifiers": identifiers,
            "interpolations": interpolations,
        }

    filename = os.path.splitext(os.path.basename(filename))[0]
    with open(f"separated-{filename}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parse_prompts("completions.txt")
    parse_prompts("chat-completions.txt")
    parse_prompts("langchain-tools.txt")
    parse_prompts("langchain-prompts.txt")
    parse_prompts("cohere-prompts.txt")
    parse_prompts("prompt_or_template_strings.txt")
