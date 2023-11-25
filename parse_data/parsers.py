import os
from tree_sitter import Language, Parser, Tree, Node
from tqdm import tqdm

if not os.path.exists("build/my-languages.so"):
    Language.build_library("build/my-languages.so", ["tree-sitter-python"])

PY_LANGUAGE = Language("./build/my-languages.so", "python")


def new_line_in_string(tree: Tree):
    """New Line in String definition heuristic"""
    result = []

    var_def_query = PY_LANGUAGE.query(
        """(expression_statement
            (assignment
                left: (identifier)
                right: (string) @var.value
            )
        )"""
    )

    for usage in var_def_query.captures(tree.root_node):
        # heuristic, check if string has a newline in it, if so then it's probably a prompt
        res = usage[0].text.decode("utf-8")
        if "\n" in res:
            result.append(res)

    return result


def prompt_or_template_in_name(tree: Tree):
    """Look for prompt or template in variable name"""
    result = []

    query = PY_LANGUAGE.query(
        """(expression_statement
            (assignment
                left: (identifier) @var.name
                right: (string) @var.value
            )
        )"""
    )

    found = False
    for usage in query.captures(tree.root_node):
        if usage[1] == "var.name":
            var_name = usage[0].text.decode("utf-8").lower()
            if "prompt" in var_name or "template" in var_name:
                found = True
        elif found:
            result.append(usage[0].text.decode("utf-8"))
            found = False

    return result


def find_def(tree: Tree, name: str):
    query = PY_LANGUAGE.query(
        f"""(expression_statement
        (assignment
            left: (identifier) @var.name
            right: (_) @var.value
            (#eq? @var.name "{name}")
        ))"""
    )

    texts = []
    for usage in filter(lambda x: x[1] == "var.value", query.captures(tree.root_node)):
        texts.append(usage[0].text.decode("utf-8"))

    return ";\n".join(texts)


def extract_args(tree: Tree, argument_list: Node) -> str:
    for idx, arg in enumerate(argument_list.children):
        # Grab first positional argument if it is an identifier
        if arg.type == "identifier" and idx == 1:
            return find_def(tree, arg.text.decode("utf-8"))

        # Grab first positional argument if it is a string
        if arg.type == "string" and idx == 1:
            return arg.text.decode("utf-8")

        if arg.type == "keyword_argument":
            return parse_keyword_argument(tree, arg)

    return argument_list.text.decode("utf-8")


def parse_keyword_argument(tree: Tree, arg: Node):
    query = PY_LANGUAGE.query(
        """(keyword_argument
                name: (identifier) @arg.name
                value: (_) @arg.value
                (#any-of? @arg.name "prompt" "messages")
        )"""
    )
    for usage in query.captures(arg):
        if usage[1] == "arg.value":
            if usage[0].type == "string":
                return usage[0].text.decode("utf-8")

            # try to find definition if ident
            return find_def(tree, usage[0].text.decode("utf-8"))

    # Default, return text
    return find_def(tree, arg.text.decode("utf-8"))


def used_in_langchain_llm_call(tree: Tree):
    """Find variables used in langchain llm calls"""
    result = []

    query = PY_LANGUAGE.query(
        """(module
                (import_from_statement
                    module_name: (dotted_name) @mod
                    name: (dotted_name) @llm
                    (#any-of? @mod "langchain.llms" "langchain.chat" "langchain")
                )
                (expression_statement
                    (assignment
                        left: (identifier) @llmvar
                        right: (call function: (identifier) @llmname)
                        (#eq? @llmname @llm)
                    )
            )
        )"""
    )

    for llm in filter(lambda x: x[1] == "llmvar", query.captures(tree.root_node)):
        llm_text = llm[0].text.decode("utf-8")

        call_query = PY_LANGUAGE.query(
            f"""(call
                function: (identifier) @fn.name
                arguments: (argument_list) @fn.args
                (#eq? @fn.name "{llm_text}")
            )"""
        )

        for usage in call_query.captures(tree.root_node):
            if usage[1] == "fn.args":
                result.append(extract_args(tree, usage[0]))

    return result


def used_in_openai_call(tree: Tree):
    """Find native openai library calls"""
    result = []

    query = PY_LANGUAGE.query(
        """(module
                (import_from_statement
                    module_name: (dotted_name) @mod
                    name: (dotted_name) @llm
                    (#eq? @mod "openai")
                )
                (expression_statement
                    (assignment
                        left: (identifier) @llmvar
                        right: (call function: (identifier) @llmname)
                        (#eq? @llmname @llm)
                    )
            )
        )"""
    )

    for llm in filter(lambda x: x[1] == "llmvar", query.captures(tree.root_node)):
        llm_text = llm[0].text.decode("utf-8")

        call_query = PY_LANGUAGE.query(
            f"""(call
                function: (attribute) @fn.name
                arguments: (argument_list) @fn.args
                (#match? @fn.name "^{llm_text}(.chat)?.completions.create")
            )"""
        )

        for usage in call_query.captures(tree.root_node):
            if usage[1] == "fn.args":
                result.append(extract_args(tree, usage[0]))

    return result


class PromptDetector:
    def __init__(self):
        self.parser = Parser()
        self.parser.set_language(PY_LANGUAGE)

        self.heuristics = []

    def add_heuristic(self, heuristic):
        self.heuristics.append(heuristic)

    def detect_prompts(self, filenames: list[str]):
        results = {}
        for filename in tqdm(filenames):
            results |= self._detect_prompts(filename)

        results = self.print_results(results)
        return {"prompts": results}

    def print_results(self, results):
        per_heuristic = {}
        prompts = set()
        count = 0
        for _, _results in results.items():
            for heuristic, found in _results.items():
                if heuristic not in per_heuristic:
                    per_heuristic[heuristic] = []

                per_heuristic[heuristic].extend(found)
                prompts.update(set(found))
                if len(found) > 0:
                    count += 1

        for heuristic in self.heuristics:
            print(heuristic.__name__)
            print(heuristic.__doc__)
            print("Found: ", len(per_heuristic[heuristic.__name__]))
            print()

        print(f"Parser Returns result for {count} files out of 1444 files")
        return list(prompts)

    def _detect_prompts(self, filename: str):
        with open(filename, "rb") as f:
            tree = self.parser.parse(f.read())

        results = {}
        for heuristic in self.heuristics:
            try:
                results[heuristic.__name__] = heuristic(tree)
            except Exception as e:
                print(f"Error in {heuristic.__name__} for {filename}")
                print(e)

        return {filename: results}
