
import sys
print(sys.argv[1])
import ast
import openai

# # Read input from command-line argument
# input_argument = sys.argv[1]


# import ast
# import openai

# # Your OpenAI GPT-3 API key
# api_key = "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"

# # Define a Python code snippet that you want to document
# code_snippet = input_argument

# # Parse the code snippet into an AST
# parsed_code = ast.parse(code_snippet)

# # Extract the relevant code information (e.g., function name, docstring)
# code_info = ""
# for node in ast.walk(parsed_code):
#     if isinstance(node, ast.FunctionDef):
#         code_info += f"Function: {node.name}\n"
#         if node.body:
#             for statement in node.body:
#                 if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Str):
#                     code_info += f"Description: {statement.value.s}\n"
#                     print(code_info)


# # Prompt for overall implementation
# prompt = f"Generate a function and method documentation of the following code:\n\n{code_info}\n\nDocumentation:"

# openai.api_key = api_key

# # Generate overall documentation using GPT-3
# response = openai.Completion.create(
#     engine="text-davinci-002",  # Choose the appropriate GPT-3 model
#     prompt=prompt,
#     max_tokens=100,  # Adjust the length as needed
# )
# generated_documentation = response.choices[0].text
# print("Overall documentation",generated_documentation)



# def extract_functions_from_code(code):
#     # Parse the input code into an abstract syntax tree (AST).
#     parsed_code = ast.parse(code)

#     # Create a list to store extracted function nodes.
#     function_nodes = []
#     def process_function(node):
#         if isinstance(node, ast.FunctionDef):
#             function_nodes.append(node)
#     for node in ast.walk(parsed_code):
#         process_function(node)

#     return function_nodes


# extracted_functions = extract_functions_from_code(code_snippet)

# prompt_full = f"""Document the code that has the following functions doing the following tasks: {generated_documentation} 
# Produce a documentation explaining the use of the function given below and its implementation in high verbosity: {extracted_functions[0]} """

# response_func = openai.Completion.create(
#     engine="text-davinci-002",  # Choose the appropriate GPT-3 model
#     prompt=prompt,
#     max_tokens=100,  # Adjust the length as needed
# )
# func_documentation = response_func.choices[0].text
# print(f"Function {extracted_functions[0]}",func_documentation)
    



# # prompt2 = f"""Document the {method_name} method of the {class_name} class.

# # Method Description:
# # {method_description}

# # Parameters:
# # - {param_name}: {param_description}

# # Return Value:
# # {return_value_description}

# # Example Usage:
# # """