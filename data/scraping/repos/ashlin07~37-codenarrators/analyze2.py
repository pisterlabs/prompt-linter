import ast
import openai

# Your OpenAI GPT-3 API key
api_key = "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"

# Define a Python code snippet that you want to document
code_snippet = """
def calculate_square(x):
    return x ** 2
"""

# Parse the code snippet into an AST
parsed_code = ast.parse(code_snippet)

# Extract the relevant code information (e.g., function name, docstring)
code_info = ""
for node in ast.walk(parsed_code):
    if isinstance(node, ast.FunctionDef):
        code_info += f"Function: {node.name}\n"
        if node.body:
            for statement in node.body:
                if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Str):
                    code_info += f"Description: {statement.value.s}\n"

# Prompt for GPT-3
prompt = f"Document the following code:\n\n{code_info}\n\nDocumentation:"

# Initialize the OpenAI API client
openai.api_key = api_key

# Generate documentation using GPT-3
response = openai.Completion.create(
    engine="text-davinci-002",  # Choose the appropriate GPT-3 model
    prompt=prompt,
    max_tokens=100,  # Adjust the length as needed
)

# Extract the generated documentation from the response
generated_documentation = response.choices[0].text

# Print or store the generated documentation
print(generated_documentation)
