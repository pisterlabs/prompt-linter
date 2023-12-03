import json
import re

from tree_sitter import Language, Parser, Tree, Node

PY_LANGUAGE = Language("./build/my-languages.so", "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)


def parse_kwarg(tree: Node):
    splits = tree.text.decode("utf-8").split("=", maxsplit=1)
    if len(splits) == 2:
        return splits[0].strip(), splits[1].strip()

    return None


def parse_args(tree: Tree):
    text = tree.root_node.text.decode("utf-8")
    fn_name = text.split("(", maxsplit=1)[0].strip()

    query = PY_LANGUAGE.query(
        f"""(call 
            function: (_) @fn
            arguments: (argument_list
                (_) @arg
            )
            (#eq? @fn "{fn_name}")
        )"""
    )

    args = {}
    for pos, arg in enumerate(query.captures(tree.root_node)):
        if arg[1] == "fn":
            continue

        if arg[0].type == "comment":
            continue

        if kwarg := parse_kwarg(arg[0]):
            args[kwarg[0]] = kwarg[1]
        else:
            args[f"pos-{pos:02d}"] = arg[0].text.decode("utf-8")

    return args


def get_parsed_files():
    with open("prompt_or_template_in_name-strings.json") as f:
        calls = json.load(f)
    return calls


def save_for_black(calls):
    import os

    os.makedirs("black7", exist_ok=True)
    for filename, el in calls:
        filename = filename.replace("/", "_")
        _calls = el["prompt_or_template_in_name"]
        for idx, call in enumerate(_calls):
            with open(f"black7/{filename}-{idx}.py", "w") as f:
                f.write("fn(" + call + ")")

    os.system("black black7/")


def get_black_trees():
    import glob

    calls = []
    files = glob.glob("black7/*.py")
    for file in files:
        with open(file, "rb") as f:
            tree = parser.parse(f.read())
            calls.append((file, tree))
    return calls


# save_for_black(get_parsed_files())
calls = get_black_trees()
call_strs = set()
all_calls = []
dups = set()
call_args = {}
for filename, tree in calls:
    call_text = tree.root_node.text.decode("utf-8")
    args = parse_args(tree)
    for key, value in args.items():
        if key not in call_args:
            call_args[key] = set()
        call_args[key].add(value)

    call_text = re.sub("\s+", " ", call_text)
    if call_text in call_strs:
        dups.add(call_text)
    call_strs.add(call_text)
    all_calls.append(call_text)

print(len(call_strs))
print(len(all_calls))
print(len(dups))


# for key, value in sorted(call_args.items()):
#     print("-----")
#     print(key, len(value))
#     print("\n".join(sorted(list(value))))

#### Below is for cohere, i.e. .chat .summarize
# with open("cohere-prompts.txt", "w") as f:
#     f.write("\n----------\n".join(call_args["text"]))
#     f.write("\n----------\n")
#     f.write("\n----------\n".join(call_args["pos-01"]))
#     f.write("\n----------\n")
#     f.write("\n----------\n".join(call_args["message"]))

#### Below is for langchain, i.e. PromptTemplate Message classes
# langchain_texts = [
#     "message",
#     "prompt",
#     "template",
#     "pos-01",
#     "pos-02",
#     "pos-03",
#     "content",
#     "prefix",
#     "suffix",
# ]
# with open("langchain-prompts.txt", "w") as f:
#     for key in langchain_texts:
#         f.write("\n----------\n".join(list(call_args[key])))
#         f.write("\n----------\n")

# with open("langchain-tools.txt", "w") as f:
#     f.write("\n----------\n".join(list(call_args["pos-01"])))


#### Below is for openai + anthropic, i.e. completions.create
# with open("openai-prompts.txt", "w") as f:
#     f.write("\n----------\n".join(call_args["prompt"]))

# with open("openai-messages.txt", "w") as f:
#     f.write("\n----------\n".join(call_args["messages"]))


with open("prompt_or_template_strings.txt", "w") as f:
    for key, value in call_args.items():
        f.write("\n----------\n".join(list(value)))
        f.write("\n----------\n")
