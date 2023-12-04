import os
import re
import json

from tree_sitter import Language, Parser, Tree

PY_LANGUAGE = Language("./build/my-languages.so", "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)
SEP = "\n----------------------------------------------------------------------------------------\n"


def parse_tree(tree: Tree):
    strings = []
    identifiers = []
    interpolations = []

    query = PY_LANGUAGE.query("(string) @string")
    for string in query.captures(tree.root_node):
        if not re.search(r"[\s\\n]", string[0].text.decode("utf-8")):
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
        prompts = f.read().split(SEP)

    prompts = list(map(lambda x: x.strip(), prompts))
    prompts = list(filter(lambda x: x != "", prompts))

    print("Found", len(prompts), "prompts in", filename)

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


def parse_metadata_file(filename):
    with open(filename) as f:
        metadata = json.load(f)

    data = {}
    clean_data = {}
    for key, value in metadata.items():
        data[key] = {}
        clean_data[key] = []
        for _, prompts in value.items():
            for prompt in prompts:
                tree = parser.parse(bytes(prompt, "utf-8"))
                strings, identifiers, interpolations = parse_tree(tree)
                data[key][prompt] = {
                    "strings": strings,
                    "identifiers": identifiers,
                    "interpolations": interpolations,
                }
                clean_data[key].extend(strings)

    filename = os.path.splitext(os.path.basename(filename))[0]
    with open(f"separated-{filename}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with open(f"clean-{filename}.json", "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parse_prompts("chat-completions.txt")
    parse_prompts("chat-completions-beta.txt")

    parse_prompts("langchain-prompts.txt")
    parse_prompts("langchain-prompts-beta.txt")
    parse_prompts("cohere-prompts.txt")
    parse_prompts("cohere-prompts-beta.txt")

    # parse_prompts("langchain-tools.txt")
    # parse_prompts("langchain-tools-class.txt")

    # parse_prompts("prompt_or_template_strings.txt")
    # parse_metadata_file("reader_prompt_metadata.json")
    # parse_metadata_file("reader_prompt_metadata_plus.json")
