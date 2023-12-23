import os
import glob
import json

from tree_sitter import Language, Parser, Tree, Node

PY_LANGUAGE = Language("./build/my-languages.so", "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)


per_file_usage = {}
all_per_file_usage = {}


def parse_kwarg(tree: Node):
    splits = tree.text.decode("utf-8").split("=", maxsplit=1)
    if len(splits) == 2:
        text = splits[1].strip()
        query = PY_LANGUAGE.query(
            """(module 
                (expression_statement 
                    (list 
                        (dictionary 
                            (pair
                                key: (_) @key
                                value: (_) @val 
                                (#match? @key "content")
                            )
                        )
                    )
                )
            )"""
        )
        texts = []
        tree2 = parser.parse(bytes(text, "utf-8"))
        captures = query.captures(tree2.root_node)
        for capture, name in captures:
            if name == "val" and capture.type == "string":
                texts.append(capture.text.decode("utf-8").strip())

        if texts == [] and captures == []:
            texts.append(text)
        return splits[0].strip(), texts

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
    for pos, (arg, aname) in enumerate(query.captures(tree.root_node)):
        if aname == "fn":
            continue

        if arg.type == "comment":
            continue

        if kwarg := parse_kwarg(arg):
            for idx, k in enumerate(kwarg[1]):
                if k.strip() != "":
                    args[f"{kwarg[0]}-{idx:02d}"] = k
        else:
            args[f"pos-{pos:02d}"] = arg.text.decode("utf-8")

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
    per_file_usage = {}
    name = os.path.splitext(os.path.basename(filename))[0]

    # with open(filename) as f:
    #     hits = json.load(f)

    # save_for_black(name, hits, wrap, wrap_extra)
    calls = get_black_trees(name)

    # call_args: dict[str, set] = {}
    for filename, args in calls:
        # if filename not in all_per_file_usage:
        #     all_per_file_usage[filename] = {}

        if filename not in per_file_usage:
            per_file_usage[filename] = {}

        for key, value in args.items():
            if key not in per_file_usage[filename] and (
                keys is None or any(k in key for k in keys)
            ):
                per_file_usage[filename][key] = []

            # if key not in all_per_file_usage[filename]:
            #     all_per_file_usage[filename][key] = []

            if keys is None or any(k in key for k in keys):
                per_file_usage[filename][key].append(value)
            # all_per_file_usage[filename][key].append(value)

    with open(f"grouped-{name}.json", "w") as f:
        ls = {}
        for key, value in per_file_usage.items():
            ls[key] = [x for y in value.values() for x in y]
        json.dump(ls, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # run("used_in_openai_call_sub.json", ["prompt", "messages"])
    # run("used_chat_function_sub.json", ["text", "pos-01", "message"])

    # keys = [
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
    # run("used_in_langchain_llm_call_sub.json", keys, wrap=False)
    # run("used_langchain_tool_class.json", ["pos-01"], wrap_extra=True)
    # run("used_langchain_tool.json", ["pos-01"], wrap_extra=True)

    # with open("reader_prompt_metadata.json", "w") as f:
    #     json.dump(per_file_usage, f, indent=2, ensure_ascii=False)

    # with open("reader_all_metadata.json", "w") as f:
    #     json.dump(all_per_file_usage, f, indent=2, ensure_ascii=False)

    # # All prompts are messier
    run("used_prompt_or_template_name.json", wrap_extra=True)

    # with open("reader_prompt_metadata_plus.json", "w") as f:
    #     json.dump(per_file_usage, f, indent=2, ensure_ascii=False)

    # with open("reader_all_metadata_plus.json", "w") as f:
    #     json.dump(all_per_file_usage, f, indent=2, ensure_ascii=False)
