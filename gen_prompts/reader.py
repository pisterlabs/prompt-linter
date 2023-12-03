import os
import glob
import json

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


def get_parsed_files(filename):
    with open(filename) as f:
        calls = json.load(f)
    return calls


def save_for_black(name, calls, wrap):
    folder = f"black-{name}"
    os.makedirs(folder, exist_ok=True)
    for filename, el in calls:
        filename = filename.replace("/", "_")
        _calls = el[name]
        for idx, call in enumerate(_calls):
            with open(f"{folder}/{filename}-{idx}.py", "w") as f:
                if wrap:
                    f.write("fn(" + call + ")")
                else:
                    f.write(call)

    os.system(folder)


def get_black_trees(name):
    calls = []
    folder = f"black-{name}"
    files = glob.glob(f"{folder}/*.py")
    for file in files:
        with open(file, "rb") as f:
            tree = parser.parse(f.read())
            calls.append((file, tree))
    return calls


def run(filename, wrap):
    name = os.path.splitext(os.path.basename(filename))[0]

    save_for_black(name, get_parsed_files(filename), wrap)
    calls = get_black_trees(name)

    call_args: dict[str, set] = {}
    for _, tree in calls:
        for key, value in parse_args(tree).items():
            if key not in call_args:
                call_args[key] = set()
            call_args[key].add(value)

    return call_args


if __name__ == "__main__":
    call_args = run("used_in_openai_call.json", wrap=True)

    ### Below is for openai + anthropic, i.e. completions.create
    with open("completions.txt", "w") as f:
        f.write("\n----------\n".join(call_args["prompt"]))

    with open("chat-completions.txt", "w") as f:
        f.write("\n----------\n".join(call_args["messages"]))

    call_args = run("used_chat_function.json", wrap=True)

    ### Below is for cohere, i.e. .chat .summarize
    cohere_texts = ["text", "pos-01", "message"]
    with open("cohere-prompts.txt", "w") as f:
        for key in cohere_texts:
            f.write("\n----------\n".join(call_args[key]))
            f.write("\n----------\n")

    call_args = run("used_langchain_llm_call.json", wrap=False)

    ### Below is for langchain, i.e. PromptTemplate Message classes
    langchain_texts = [
        "message",
        "prompt",
        "template",
        "pos-01",
        "pos-02",
        "pos-03",
        "content",
        "prefix",
        "suffix",
    ]
    with open("langchain-prompts.txt", "w") as f:
        for key in langchain_texts:
            f.write("\n----------\n".join(list(call_args[key])))
            f.write("\n----------\n")

    call_args = run("used_langchain_tool.json", wrap=True)

    with open("langchain-tools.txt", "w") as f:
        f.write("\n----------\n".join(list(call_args["pos-01"])))

    call_args = run("prompt_or_template_in_name.json", wrap=True)

    with open("prompt_or_template_strings.txt", "w") as f:
        for key, value in call_args.items():
            f.write("\n----------\n".join(list(value)))
