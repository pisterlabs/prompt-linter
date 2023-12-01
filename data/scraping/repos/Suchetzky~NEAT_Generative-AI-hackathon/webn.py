import io

import streamlit as st
import time
import pandas as pd

import networkx as nx
import json
import subprocess
import re
import ast
import openai


# All scripts ##########################
class FunctionImprovements:

    def __init__(self, old_code, new_code, comment):
        self.old_code = old_code
        self.new_code = new_code
        self.comment = comment


class GPT:
    def __init__(self, model="gpt-3.5-turbo", max_tokens=2048):
        self.model_ = model
        self.max_tokens_ = max_tokens
        openai.organization = "org-W1xrGR4WAmmdeqk5vOBvlntj"
        #Add your Api key here and uncomment
        #openai.api_key = 
        self.messages_ = []

    # DEAL WITH CHANGES TO MODEL
    def switch_model(self, model):
        """
        Switches the model to a new model
        """
        self.model_ = model

    @staticmethod
    def get_models():
        """
        :return: A list of all available models to use.
        """
        return openai.Model.list()

    # DEAL WITH CHANGES TO MESSAGE, SYSTEM
    def add_system_message(self, content):
        self.messages_.append({"role": "system", "content": content})

    def replace_system_message(self, content):
        self.messages_[0] = {"role": "system", "content": content}

    # DEAL WITH CHANGES TO MESSAGES, USER AND ASSISTANT

    def remove_first_k_messages(self, k):
        """
        Removes the first k messages from the messages list not including the system message
        """
        self.messages_ = self.messages_[0] + self.messages_[k:]

    def clear_messages(self):
        """
        Clears the messages list
        """
        self.messages_ = [self.messages_[0]]

    def chat(self, content):
        """

        :param content:
        :return:
        """
        self.messages_.append({"role": "user", "content": content})
        response = openai.ChatCompletion.create(model=self.model_, messages=self.messages_, temperature=0,
                                                max_tokens=self.max_tokens_)
        assistant_msg = response['choices'][0]['message']['content']
        self.messages_.append({"role": "assistant", "content": assistant_msg})
        return assistant_msg


STYLE_PROMPTS_FILE = "lib/prompts/style_prompts.json"
TEST_PROMPTS_FILE = "lib/prompts/test_prompts.json"
TEST_CODE_FILE = "tests.py"
MY_MODULE_FILE = "mymodule.py"


class TestResponse:
    def __init__(self, name, source_code, error=None, explanation=None):
        self.name = name
        self.source_code = source_code
        self.error = error
        self.explanation = explanation


def generate_responses(test_names, test_contents, errors):
    responses = []
    for name, content, error in zip(test_names, test_contents, errors):
        responses.append(TestResponse(name, content, error))
    return responses


def responseFromText(tests, errors):
    test_names, test_contents = extract_test_functions(tests)
    test_names = [function.name for function in test_names]
    errors_dict = extract_errors(errors)
    errors = [None] * len(test_names)
    for test_name, test_content in errors_dict.items():
        errors[test_names.index(test_name)] = test_content
    return generate_responses(test_names, test_contents, errors)


def extract_errors(source):
    errors = dict()
    row_pattern = "(FAIL|ERROR):[^-]*"
    function_name_pattern = "(FAIL|ERROR):(.*)\("
    for match in re.finditer(row_pattern, source):
        full_text = match.group(0)
        func_name = list(re.finditer(function_name_pattern, full_text))[0].group(2).strip()
        errors[func_name] = full_text
    return errors


def extract_test_functions(source_code):
    function_sources = []
    ast_tree = ast.parse(source_code)
    unit_tests_class = [item for item in ast_tree.body if type(item) == ast.ClassDef][0]
    function_names = extract_functions(unit_tests_class)
    for function in function_names:
        function_source = get_function_source(source_code=source_code, tree=unit_tests_class,
                                              function_name=function.name)
        function_sources.append(function_source)
    return function_names, function_sources


def parse_base_response(response):
    pattern = "```"
    regex = re.compile(pattern)
    matches = list(regex.finditer(response))
    if len(matches) == 0:
        return response
    response = response[matches[0].end(): matches[1].start()]
    if not response.startswith("python"):
        response = response[6:]
    return response


def get_test_suggestions(server_side, source_code):
    """Generate tests for the python function given in source-code,
    where the documentations of the functions it depends on are listed in
    'dependencies_documentations', and are assumed to be valid and working. """

    with open(TEST_PROMPTS_FILE) as f:
        prompts = json.load(fp=f)
    with open(MY_MODULE_FILE, 'w') as f:
        f.write(source_code)

    gpt = GPT()
    gpt.add_system_message(prompts.get('base_prompt')[0])
    # function_codes_generator = server_side.get_sources()
    # funcResponses = []
    # for function_code in tqdm(function_codes_generator):
    #     GPT.clear_messages()
    #     response = GPT.chat(function_code)
    #     parsed_response = parse_base_response(response)
    #     with open(TEST_CODE_FILE, 'w') as tests_file:
    #         tests_file.write(parsed_response)
    #     p = subprocess.Popen(f"python3 {TEST_CODE_FILE}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    #     _, err = p.communicate()
    #     error = err.decode('utf-8')
    #     function_response = FunctionResponse.fromText(parsed_response, error)
    #     funcResponses.append(function_response)
    funcResponse = None
    response = gpt.chat(source_code)
    parsed_response = parse_base_response(response)
    with open(TEST_CODE_FILE, 'w') as tests_file:
        tests_file.write(parsed_response)
    p = subprocess.Popen(f"python3 {TEST_CODE_FILE}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    _, err = p.communicate()
    error = err.decode('utf-8')
    function_response = responseFromText(parsed_response, error)
    return function_response


def get_function_source(source_code, tree, function_name):
    # Get the start and end line numbers of the function definition
    function_node = None
    function_index = 0
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            function_node = node
            function_index = index
            break
    if not function_node:
        return None

    start_line = function_node.lineno
    # Extract the lines containing the function
    lines = source_code.split('\n')

    if function_index == len(tree.body) - 1:
        function_lines = lines[start_line - 1:]
    else:
        function_lines = lines[start_line - 1: tree.body[function_index + 1].lineno - 1]
    # Join the lines back into a string
    function_source = '\n'.join(function_lines)
    return function_source


def parse_base_response_style(response):
    pattern = "```"
    regex = re.compile(pattern)
    matches = list(regex.finditer(response))
    code = response[matches[0].end(): matches[1].start()]
    if response.startswith("python"):
        code = code[6:]
    return code, response[matches[1].end():]


def get_style_suggestions(server_side):
    with open(STYLE_PROMPTS_FILE) as f:
        prompts = json.load(fp=f)
    gpt = GPT()
    gpt.add_system_message(prompts.get('base_prompt')[0])

    suggestions = {f: [] for f in server_side.functions_}
    for func in suggestions.keys():
        gpt.clear_messages()
        old_code = server_side.get_function_source(func)
        response = gpt.chat(server_side.get_function_source(func))
        new_code, explanation = parse_base_response_style(response)
        suggestions[func] = FunctionImprovements(old_code, new_code, explanation)
    return suggestions


def extract_functions(node):
    """
    Extracts all function definitions from an AST node.
    """
    functions = []
    for body_item in node.body:
        if isinstance(body_item, ast.FunctionDef):
            functions.append(body_item)
    return functions


def extract_function_calls(node):
    """
    Extracts all function calls from an AST node.
    """
    function_calls = []
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function_calls.append(node.func.id)
    for child_node in ast.iter_child_nodes(node):
        function_calls.extend(extract_function_calls(child_node))
    return function_calls


def build_dependency_graph(functions):
    """
    Builds a dependency graph of functions in a Python code file.
    """
    graph = nx.DiGraph()
    for function in functions:
        function_name = function.name
        graph.add_node(function_name)
        function_calls = extract_function_calls(function)
        for called_function in function_calls:
            if function_name == called_function:
                continue
            if called_function not in functions:
                continue
            graph.add_edge(function_name, called_function)
    return graph


def build_topological_sort(source_code):
    """Returns a list of functions sorted topologically"""
    ast_tree = ast.parse(source_code)
    functions = extract_functions(ast_tree)
    graph = build_dependency_graph(functions)
    sorted_functions = list(nx.topological_sort(graph))
    return sorted_functions


class ServerSide:
    def __init__(self, code=None):
        if code is not None:
            self.code_ = code.decode('utf-8')
            self.functions_\
                = build_topological_sort(code)
            self.ast_tree_ = ast.parse(code)

    def set_code(self, code):
        self.code_ = code.decode('utf-8')
        self.functions_ = build_topological_sort(code)
        self.ast_tree_ = ast.parse(code)

    def get_style_suggestions(self):
        """
        The function will recieve a python code string and return a list of suggestions based on the preferences
            specified.
        :return: A dictionary of suggestions, each key is a function name and the value is a FunctionImprovements object
        """
        if self.code_ is None:
            raise Exception("Code is not set, please call set_code first")

        return get_style_suggestions(self)

    def get_tests_suggestions(self):
        """
        The function will recieve a python code string and return a list of suggestions based on the preferences
            specified.
        :return: A list of tests run results
        """
        if self.code_ is None:
            raise Exception("Code is not set, please call set_code first")

        return get_test_suggestions(self, self.code_)

    def get_function_source(self, function_name):
        # Get the start and end line numbers of the function definition
        function_node = None
        function_index = 0
        for index, node in enumerate(self.ast_tree_.body):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                function_node = node
                function_index = index
                break
        if not function_node:
            return None

        start_line = function_node.lineno
        # Extract the lines containing the function
        print(type(self.code_))
        lines = self.code_.split('\n')

        if function_index == len(self.ast_tree_.body) - 1:
            function_lines = lines[start_line - 1:]
        else:
            function_lines = lines[start_line - 1: self.ast_tree_.body[function_index + 1].lineno - 1]
        # Join the lines back into a string
        function_source = '\n'.join(function_lines)
        return function_source

    def get_sources(self):
        """
        Gets source code and returns source codes of functions by topological sort order.
        :return:
        """
        for function_name in reversed(self.functions_):
            function_source = self.get_function_source(function_name)
            yield function_source


########################################
if 'key' not in st.session_state:
    st.session_state["returned_data"] = None
    st.session_state["keys"] = None
    st.session_state["new_code_data"] = ""
    st.session_state.key = 'UploadFile'


def call_api_improve():
    return {"1": FunctionImprovements("old", "new", "your code is bad"),
            "2": FunctionImprovements("old1", "new2", "your code is bad2")}


def call_api_test():
    return [TestResponse("testfffffffffffffff1", "cofffffsssde1", "errossssssssssssssr1", "explansssssation1"),
            TestResponse("test2", "code2", None, "explanation2")]


def run_improve():
    loading = st.empty()
    loading.write('Loading...')
    time.sleep(2)
    st.session_state["returned_data"] = st.session_state['server_side'].get_style_suggestions()
    st.session_state["keys"] = list(st.session_state["returned_data"].keys())
    loading.empty()
    st.session_state.key = 'Improve'


def run_testing():
    loading = st.empty()
    loading.write('Loading...')
    time.sleep(2)
    st.session_state["returned_data"] = st.session_state['server_side'].get_tests_suggestions()
    loading.empty()
    st.session_state.key = 'TestResults'


st.title("Review :blue[Your] Code")

if st.session_state.key == 'UploadFile':
    ch_text = st.empty()
    ch_text.write("First We need to see your code")
    ch_upload = st.empty()
    code_data = ch_upload.file_uploader("Upload Code")
    if code_data:
        st.session_state['server_side']=ServerSide(code_data.read())
        st.session_state.key = 'ChooseOperation'
        ch_text.empty()
        ch_upload.empty()

if st.session_state.key == 'ChooseOperation':

    # Create an empty placeholder for the start button
    ch_improve_button = st.empty()
    ch_test_button = st.empty()

    # Create the start button and assign the 'start' function to its 'on_click' parameter
    improve_button = ch_improve_button.button("Improve My Code")
    test_button = ch_test_button.button("Test My Code")

    # If the start button is clicked, clear the placeholders
    if improve_button or test_button:
        ch_improve_button.empty()
        ch_test_button.empty()
        st.session_state.key = 'loading'
        if improve_button:
            run_improve()
            st.session_state.count = 0
        else:
            run_testing()

if st.session_state.key == 'runningApi':
    # Define the initial code string
    pass

if st.session_state.key == 'Improve':
    ch_user_code = st.empty()
    ch_improved_code = st.empty()
    ch_comment_code = st.empty()
    returned_data = st.session_state["returned_data"]
    keys = st.session_state["keys"]
    # Create empty placeholders for the 'accept yours' and 'accept theirs' buttons
    ch_accept_yours = st.empty()
    ch_accept_theirs = st.empty()

    # Create the 'accept yours' and 'accept theirs' buttons
    accept_yours = ch_accept_yours.button("Accept Yours")
    accept_theirs = ch_accept_theirs.button("Accept Theirs")
    if accept_yours:
        # ch_accept_yours.empty()
        # ch_accept_theirs.empty()
        st.session_state["new_code_data"] += returned_data[keys[st.session_state.count]].old_code + "\n"
        st.session_state.count += 1
        # if st.session_state.count == len(keys):
        #     #todo remove everything and add download button to new code
        #     st.session_state.key = 'finished'
    if accept_theirs:
        st.session_state["new_code_data"] += returned_data[keys[st.session_state.count]].new_code + "\n"
        st.session_state.count += 1
        #     #todo remove everything and add download button to new code
        #     st.session_state.key = 'finished'
    if st.session_state.count == len(keys):
        ch_accept_yours.empty()
        ch_accept_theirs.empty()
        ch_improved_code.empty()
        ch_comment_code.empty()
        ch_user_code.empty()
        st.session_state.key = "Download"

    else:
        print("i is equals to " + str(st.session_state.count))
        user_code = ch_user_code.code(returned_data[keys[st.session_state.count]].old_code, language='python',
                                      line_numbers=True)
        # Display the user's code with syntax highlighting
        improved_code = ch_improved_code.code(returned_data[keys[st.session_state.count]].new_code, language='python',
                                              line_numbers=True)
        explanation = ch_comment_code.code(returned_data[keys[st.session_state.count]].comment, language='python',
                                           line_numbers=True)

if st.session_state.key == 'TestResults':

    to_add = []
    for i in st.session_state["returned_data"]:
        status = "✔️"
        if i.error is not None:
            status = "❌"
        to_add.append(
            {"Test name": i.name, "Test": i.source_code, "Error": i.error, "Description (Expendable)": i.explanation,
             "Status": status})
    df = pd.DataFrame(
        to_add
    )
    edited_df = st.experimental_data_editor(df)

if st.session_state.key == "Download":
    st.download_button('Download Result', st.session_state["new_code_data"])
