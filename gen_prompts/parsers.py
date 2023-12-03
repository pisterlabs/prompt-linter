import json
import os
from tree_sitter import Language, Parser, Tree, Node
from tqdm import tqdm

if not os.path.exists("build/my-languages.so"):
    Language.build_library("build/my-languages.so", ["vendor/tree-sitter-python"])

PY_LANGUAGE = Language("./build/my-languages.so", "python")

# Parsers should be of type: Tree -> list[prompt_dict]
# prompt_dict should be a dictionary with a prompt key and a metadata key


def all_strings(tree: Tree):
    """All Strings heuristic"""
    result = []

    query = PY_LANGUAGE.query("((string) @var.value)")

    for usage in query.captures(tree.root_node):
        string = usage[0].text.decode("utf-8")
        if string.count(" ") > 2:
            result.append({"string": usage[0].text.decode("utf-8"), "metadata": {}})

    return result


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
            result.append({"prompt": res, "metadata": {}})

    return result


def prompt_or_template_in_name(tree: Tree):
    """Look for prompt or template in variable name"""
    result = []

    query = PY_LANGUAGE.query(
        """(expression_statement
            (assignment
                left: (identifier) @var.name
                right: (_) @var.value
            )
            (#match? @var.name "([Pp][Rr][Oo][Mm][Pp][Tt]|[Tt][Ee][Mm][Pp][Ll][Aa][Tt][Ee])")
        ) @assign"""
    )

    found = False
    var_name = ""
    for usage in query.captures(tree.root_node):
        if usage[1] == "var.name":
            var_name = usage[0].text.decode("utf-8").lower()
            if (
                "prompt" in var_name
                or "template" in var_name
                or "message" in var_name
                or "content" in var_name
            ):
                found = True
        elif found:
            # TODO find each usage of this variable name?
            # result.append(
            #     {
            #         "prompt": usage[0].text.decode("utf-8"),
            #         "metadata": {"name": var_name},
            #     }
            # )
            # result[-1]["metadata"]["defs"] = find_def(tree, var_name)
            found = False
        if usage[1] == "assign":
            result.append(usage[0].text.decode("utf-8"))

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

    texts = None
    for usage in filter(lambda x: x[1] == "var.value", query.captures(tree.root_node)):
        if not texts:
            texts = str_to_prompt_dict(
                usage[0].text.decode("utf-8"), metadata={"assignments": []}
            )
        else:
            texts["metadata"]["assignments"].append(usage[0].text.decode("utf-8"))

    if texts:
        return [texts]
    return []


def parse_call_argument_generic(arg: Node):
    query = PY_LANGUAGE.query(
        """(call
            function: (identifier) @fn.name
            arguments: (argument_list) @fn.args
        )"""
    )

    result = []
    name = None
    for usage in query.captures(arg):
        if usage[1] == "fn.name":
            name = usage[0].text.decode("utf-8")
        elif usage[1] == "fn.args" and name:
            # TODO grab idents out of here
            result.append(
                {
                    "prompt": usage[0].text.decode("utf-8"),
                    "metadata": {"name": name, "generic": True},
                }
            )

    return result


def parse_call_argument(arg: Node):
    query = PY_LANGUAGE.query(
        """(call
            function: (attribute
                object: (string) @str
                attribute: (_)
            )
            arguments: (argument_list) @fn.args
        )"""
    )

    result = []
    for usage in query.captures(arg):
        if usage[1] == "str":
            result.append({"prompt": usage[0].text.decode("utf-8"), "metadata": {}})
        elif usage[1] == "fn.args":
            result[-1]["metadata"]["args"] = usage[0].text.decode("utf-8")

    return result


def parse_dictionary(tree: Tree, arg: Node):
    query = PY_LANGUAGE.query(
        """(dictionary 
                (pair
                    key: (_) @keyval
                    value: (_) @val
                )
                (#match? @keyval "content")
        )"""
    )

    result = []
    for usage in query.captures(arg):
        if usage[1] == "val":
            # TODO grab idents out of here
            result.append(str_to_prompt_dict(usage[0].text.decode("utf-8")))

    return result


def get_common(tree: Tree, arg: None):
    # Most common ways to call an llm are with a string, with a variable, with a list, or with a function call

    # String is easiest, just return it
    if arg.type == "string":
        return [str_to_prompt_dict(arg.text.decode("utf-8"))]

    # Variables search for their definition, acquiring metadata about all assignments
    # They stop at their scope
    if arg.type == "identifier":
        return find_def(tree, arg.text.decode("utf-8"))

    # Calls are specifically for `string.format`
    if arg.type == "call":
        return parse_call_argument(arg)

    # A list can be a list of variables, calls, or dictionaries
    values = []
    if arg.type == "list":
        for el in arg.children:
            if el.type == "identifier":
                values.extend(find_def(tree, el.text.decode("utf-8")))
            if el.type == "call":
                values.extend(parse_call_argument(el))
                values.extend(parse_call_argument_generic(el))
            if el.type == "dictionary":
                values.extend(parse_dictionary(tree, el))

    return values


def extract_args(tree: Tree, argument_list: Node) -> list:
    result = []
    metadata = {"extract_args": argument_list.text.decode("utf-8")}
    for idx, arg in enumerate(argument_list.children):
        # Grab first positional argument if it is an identifier
        if idx == 1:
            result.extend(get_common(tree, arg))

        if arg.type == "keyword_argument":
            result.extend(parse_keyword_argument(tree, arg))

    for res in result:
        res["metadata"] |= metadata

    return result or [
        str_to_prompt_dict(metadata={"error": argument_list.text.decode("utf-8")})
    ]


def str_to_prompt_dict(string: str = "", metadata=None):
    metadata = metadata or {}
    if isinstance(metadata, str):
        metadata = {"error": metadata}
    return {"prompt": string, "metadata": metadata}


def parse_keyword_argument(tree: Tree, arg: Node):
    query = PY_LANGUAGE.query(
        """(keyword_argument
                name: (identifier) @arg.name
                value: (_) @arg.value
                (#any-of? @arg.name "prompt" "messages" "text" "content")
        )"""
    )
    result = []
    for usage in query.captures(arg):
        if usage[1] == "arg.value":
            if common := get_common(tree, usage[0]):
                result.extend(common)
            else:
                # Technically an error case
                result.append(
                    str_to_prompt_dict(metadata={"error": arg.text.decode("utf-8")})
                )

    # Default, return text
    return result


def used_langchain_tool(tree: Tree):
    tool_query = PY_LANGUAGE.query(
        """(decorated_definition
        (decorator (identifier) @dec)
        definition: (function_definition
            name: (identifier)
            parameters: (_)
            return_type: (_)
            body: (block
                (expression_statement (string) @docstring))
        )
        (#eq? @dec "tool")
    )"""
    )
    result = []
    for capture in tool_query.captures(tree.root_node):
        if capture[1] == "docstring":
            result.append(capture[0].text.decode("utf-8"))
    return result


def used_in_langchain_llm_call(tree: Tree):
    """Find variables used in langchain llm calls"""
    result = []
    from_template_query = PY_LANGUAGE.query(
        """(call 
        function: 
        (attribute
            object: (identifier) @ob
            attribute: (identifier)
        )
        (#match? @ob "Template$")
        arguments: (argument_list)
    ) @call"""
    )

    template_query = PY_LANGUAGE.query(
        """(call 
        function: (identifier) @ob
        (#match? @ob "(Template|Message)$")
        arguments: (argument_list)
    ) @call"""
    )

    for capture in template_query.captures(tree.root_node):
        if capture[1] == "call":
            result.append(capture[0].text.decode("utf-8"))

    for capture in from_template_query.captures(tree.root_node):
        if capture[1] == "call":
            result.append(capture[0].text.decode("utf-8"))

    return result

    """llm.<fn>(<args>)"""

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


def used_chat_function(tree: Tree):
    query = PY_LANGUAGE.query(
        """(call 
        function: 
        (attribute
            object: (identifier)
            attribute: (identifier) @fn
        )
        (#match? @fn "^(chat|summarize)$")
        arguments: (argument_list)
    ) @call"""
    )

    results = []
    for capture in query.captures(tree.root_node):
        if capture[1] == "call":
            results.append(capture[0].text.decode("utf-8"))
    return results


def used_in_openai_call(tree: Tree):
    """Find native openai library calls"""
    result = []

    call_query = PY_LANGUAGE.query(
        """(call
            function: (attribute) @fn.name
            arguments: (argument_list) @fn.args
            (#match? @fn.name "(\.[Cc]hat)?\.?[cC]ompletions?\.create")
        )"""
    )

    for usage in filter(
        lambda x: x[1] == "fn.args", call_query.captures(tree.root_node)
    ):
        result.append(usage[0].text.decode("utf-8"))
        # result.extend(extract_args(tree, usage[0]))

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

        for heuristic in self.heuristics:
            name = heuristic.__name__
            results = filter(lambda x, name=name: x[1][name], results.items())

            with open(f"{name}.json", "w") as f:
                json.dump(list(results), f, indent=2)

        return []

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

        """
        for heuristic in self.heuristics:
            print(heuristic.__name__)
            print(heuristic.__doc__)
            print("Found: ", len(per_heuristic[heuristic.__name__]))
            print()

        print(f"Parser Returns result for {count} files out of 1444 files")
        """
        return list(prompts)

    def _detect_prompts(self, filename: str):
        with open(filename, "rb") as f:
            tree = self.parser.parse(f.read())

        results = {}
        for heuristic in self.heuristics:
            results[heuristic.__name__] = heuristic(tree)

        return {filename: results}
