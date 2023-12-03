import os
import glob
import json

from tree_sitter import Language, Parser, Tree, Node

PY_LANGUAGE = Language("./build/my-languages.so", "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)


per_file_usage = {}
all_per_file_usage = {}
SEP = "\n----------------------------------------------------------------------------------------\n"


def parse_kwarg(tree: Node):
    splits = tree.text.decode("utf-8").split("=", maxsplit=1)
    if len(splits) == 2:
        return splits[0].strip(), splits[1].strip()

    return None


def parse_args(tree: Tree):
    text = tree.root_node.text.decode("utf-8")
    fn_name = text.split("(", maxsplit=1)[0].strip()
    fn_name = fn_name.split("\n")[-1].strip()

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


def save_for_black(name, calls, wrap, wrap_extra):
    folder = f"black-{name}"
    os.makedirs(folder, exist_ok=True)
    for ogfilename, el in calls:
        filename = ogfilename.replace("/", "_")
        _calls = el[name]
        for idx, call in enumerate(_calls):
            with open(f"{folder}/{filename}-{idx}.py", "w") as f:
                f.write(f"# {ogfilename}\n")
                if wrap_extra:
                    f.write("fn(" + call + ")")
                elif wrap:
                    f.write("fn" + call)
                else:
                    f.write(call)

    os.system(f"black {folder} -q")


def get_black_trees(name):
    calls = []
    folder = f"black-{name}"
    files = glob.glob(f"{folder}/*.py")
    for file in files:
        with open(file) as f:
            filename = f.readline()[2:].strip()

        with open(file, "rb") as f:
            tree = parser.parse(f.read())
            calls.append((filename, parse_args(tree)))
    return calls


def run(filename, keys=None, wrap=True, wrap_extra=False):
    name = os.path.splitext(os.path.basename(filename))[0]

    with open(filename) as f:
        hits = json.load(f)

    save_for_black(name, hits, wrap, wrap_extra)
    calls = get_black_trees(name)

    call_args: dict[str, set] = {}
    for filename, args in calls:
        if filename not in all_per_file_usage:
            all_per_file_usage[filename] = {}

        if filename not in per_file_usage:
            per_file_usage[filename] = {}

        for key, value in args.items():
            if key not in call_args:
                call_args[key] = set()

            if key not in per_file_usage[filename] and (keys is None or key in keys):
                per_file_usage[filename][key] = []

            if key not in all_per_file_usage[filename]:
                all_per_file_usage[filename][key] = []

            call_args[key].add(value)
            if keys is None or key in keys:
                per_file_usage[filename][key].append(value)
            all_per_file_usage[filename][key].append(value)

    return call_args


def save_with_sep(name, call_args, keys):
    with open(f"{name}.txt", "w") as f:
        for key in keys:
            f.write(SEP.join(call_args.get(key, [])))
            f.write(SEP)


if __name__ == "__main__":
    keys = ["prompt", "messages"]
    call_args = run("used_in_openai_call.json", keys)
    save_with_sep("chat-completions", call_args, keys)

    keys = ["text", "pos-01", "message"]
    call_args = run("used_chat_function.json", keys)
    save_with_sep("cohere-prompts", call_args, keys)

    keys = [
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
    call_args = run("used_in_langchain_llm_call.json", keys, wrap=False)
    save_with_sep("langchain-prompts", call_args, keys)

    keys = ["pos-01"]
    call_args = run("used_langchain_tool.json", keys, wrap_extra=True)
    save_with_sep("langchain-tools", call_args, keys)

    with open("reader_prompt_metadata.json", "w") as f:
        json.dump(per_file_usage, f, indent=2, ensure_ascii=False)

    with open("reader_all_metadata.json", "w") as f:
        json.dump(all_per_file_usage, f, indent=2, ensure_ascii=False)

    # All prompts are messier
    call_args = run("used_prompt_or_template_name.json", wrap_extra=True)
    save_with_sep("prompt_or_template_strings", call_args, list(call_args.keys()))

    with open("reader_prompt_metadata_plus.json", "w") as f:
        json.dump(per_file_usage, f, indent=2, ensure_ascii=False)

    with open("reader_all_metadata_plus.json", "w") as f:
        json.dump(all_per_file_usage, f, indent=2, ensure_ascii=False)
