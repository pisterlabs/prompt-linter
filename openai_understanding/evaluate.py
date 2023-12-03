import json

from tree_sitter import Language, Parser, Tree

PY_LANGUAGE = Language("./build/my-languages.so", "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)

with open("prompt_or_template_strings.txt") as f:
    prompts = f.read().split("\n----------\n")
    prompts = list(map(lambda x: x.strip(), prompts))
    prompts = list(filter(lambda x: x != "", prompts))

print(len(prompts))


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


data = {}
for prompt in prompts:
    tree = parser.parse(bytes(prompt, "utf-8"))
    strings, identifiers, interpolations = parse_tree(tree)
    data[prompt] = {
        "strings": strings,
        "identifiers": identifiers,
        "interpolations": interpolations,
    }


with open("separated-name-based.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
