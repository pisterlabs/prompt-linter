import openai
import json

# Set up OpenAI API credentials
openai.api_key = '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'

def convert_to_notebook(file_name):
    # Read the content of the Python file
    with open(file_name, 'r') as file:
        python_code = file.read()

    # Define the prompt
    prompt = """
    Convert the provided Python script into the JSON source for an intuitive Jupyter Notebook that creates a user-friendly, visually appealing, and informative notebook file in the IPYNB format. Ensure that the output is reliably in JSON format (the output should be valid json), includes markdown cell data, and code cell data for every possible detail in the script that can be documented. The goal is to create a notebook that breaks the script into logical chunks and describes each section's functionality. It should clearly explain what is executing and what configurations are available for executing the script if called from the command line. The final output should be a JSON format valid structure with extensino IPYNB that serves as documentation and allows testing of the code in the script.

    ## Python Script:
    """

    # Append the Python script to the prompt
    prompt += python_code

    # Call OpenAI API to convert the prompt to a notebook
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0.3,
        max_tokens=3000,
        n=1,
        stop=None,
        timeout=300
    )

    # Extract the notebook from the response
    notebook = response.choices[0].text.strip()

    # Save the notebook as a JSON file
    notebook_file_name = file_name.replace('.py', '.ipynb')
    with open(notebook_file_name, 'w') as notebook_file:
        notebook_file.write(notebook)

    print(f"Conversion complete. The notebook is saved as {notebook_file_name}.")

# Specify the Python file to convert
file_name = 'igt.py'

# Convert the file to a notebook
convert_to_notebook(file_name)
