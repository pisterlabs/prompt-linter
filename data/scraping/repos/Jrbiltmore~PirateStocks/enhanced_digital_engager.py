import json
import openai

# Function to complete the modules list with provided module names
def complete_modules_list(module_names):
    modules = []
    for module_name in module_names:
        filename = module_name.lower().replace(" ", "_") + ".json"
        modules.append({"module_name": module_name, "filename": filename, "functions": []})
    return modules

# OpenAI API Key
OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'

# Function to generate responses from the digital-engager using the OpenAI API
def generate_response(prompt, max_tokens=150):
    user_input = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=max_tokens,
        api_key=OPENAI_API_KEY
    )
    return user_input['choices'][0]['text'].strip()

# Prompt the digital-engager for project description and handle errors
def get_project_description():
    while True:
        try:
            project_description_prompt = "Welcome to the Pirate Stocks App Project Setup! Please provide a brief description of the project:"
            project_description = generate_response(project_description_prompt)
            if project_description:
                return project_description
            else:
                print("Invalid response. Please try again.")
        except Exception as e:
            print("Error:", str(e))

# Prompt the digital-engager for module names and handle errors
def get_module_names():
    module_names = []
    try:
        module_names_prompt = "Now, enter module names for the app (type 'done' to finish):\n"
        while True:
            module_name = generate_response(module_names_prompt)
            if module_name.lower() == "done":
                break
            elif module_name:
                module_names.append(module_name)
            else:
                print("Invalid response. Please try again.")
        return module_names
    except Exception as e:
        print("Error:", str(e))

# Complete the modules list with generated filenames and function descriptions
def setup_modules(module_names):
    return complete_modules_list(module_names)

# Function to generate code for the function descriptions using the digital-engager
def generate_function_descriptions(module):
    function_descriptions = []
    for function in module["functions"]:
        function_description_prompt = f"Please provide a brief description of the function '{function['name']}' in module '{module['module_name']}':"
        function_description = generate_response(function_description_prompt)
        function_descriptions.append({"name": function["name"], "description": function_description})
    return function_descriptions

# Generate code for the function descriptions using the digital-engager
def setup_function_descriptions(modules):
    for module in modules:
        module["functions"] = generate_function_descriptions(module)
    return modules

# Create JSON templates for each module
def json_template(file_outline):
    with open(f"{file_outline['filename']}", "w") as f:
        f.write(json.dumps(file_outline, indent=4))

# Save modules as JSON templates
def save_modules(modules):
    for module in modules:
        json_template(module)

# Generate 'openai_script.py' file
def generate_openai_script(project_description, modules):
    script = f"""
import json

# Project Description
project_description = \"{project_description}\"

# List of Modules
modules = {json.dumps(modules, indent=4)}

# Rest of the code remains unchanged...
"""
    with open("openai_script.py", "w") as f:
        f.write(script)

# Main function to setup the project and generate 'openai_script.py' file
def main():
    # Setup the project
    project_description = get_project_description()
    module_names = get_module_names()
    modules = setup_modules(module_names)
    modules = setup_function_descriptions(modules)

    # Save modules as JSON templates
    save_modules(modules)

    # Generate 'openai_script.py' file
    generate_openai_script(project_description, modules)

if __name__ == "__main__":
    main()
