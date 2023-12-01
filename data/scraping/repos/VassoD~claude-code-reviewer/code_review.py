from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os
import csv
import argparse

# Get API key from environment variable
# You can request your API key from https://www.anthropic.com/earlyaccess. It's free for now.
api_key = os.getenv("ANTHROPIC_API_KEY")
if api_key is None:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

# Initialize Anthropic client
anthropic = Anthropic(api_key=api_key)

# Add the extensions of the code files you want to analyze 
EXTENSIONS = ['.py', '.js', 'ts,' ,'tsx', '.html', '.css']

# Function to analyze code with my prompts. You can change the prompts to your own needs and test the model.
def analyze_code(code, file, context):
    prompt = f"""{HUMAN_PROMPT} Please briefly review this code snippet in the context of the project:

{code}

In your response:
- List any errors or issues found in {file}
- Provide 2-3 suggestions for improvement in {file}
- Share links to relevant coding standards/documentation for {file}
- Consider the context provided by .gitignore:

{context}
If a file is ignored, don't suggest improvements related to security for the file with the same name.

Use simple language a beginner would understand

Response format for {file}:

[Issue 1]
- Suggestion 1 
- Suggestion 2

[Issue 2]
- Suggestion

Documentation:
- Link 1
- Link 2 {AI_PROMPT}"""

    response = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=500,
        prompt=prompt
    ).completion

    return response

# Function to analyze files of the specified folder
def analyze_files(folder):
    # Get all files in the folder
    files = []
    for root, dirs, filenames in os.walk(folder):
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in EXTENSIONS:
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as f:
                    code = f.read()
                files.append({'file': filepath, 'code': code})

    # Read the contents of another file so that we can provide some context(optional).
    # I used this kind of context of gitignore because I wanted to test if the model will keep telling me about my "hardcoded" API key or not
    # which I pass in the gitignore file so my API is actually not in production. The context worked. See the latest report.md file.
    context = ""
    gitignore_file = os.path.join(folder, ".gitignore")
    if os.path.exists(gitignore_file):
        with open(gitignore_file, 'r') as gitignore:
            context = gitignore.read()
    print(context)

    # Analyze all code snippets
    results = []
    for file_data in files:
        analysis = analyze_code(file_data['code'], file_data['file'], context)
        results.append({'file': file_data['file'], 'analysis': analysis})

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Analysis Script")
    parser.add_argument("folder", help="Folder to analyze")
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        raise ValueError(f"The specified folder '{args.folder}' does not exist.")

    # Analyze the user-specified folder
    results = analyze_files(args.folder)

    # Write results to a Markdown text file
    with open('report.md', 'w') as f:
        f.write("# Code Analysis Report\n\n")
        for result in results:
            f.write(f"## File: {result['file']}\n\n")
            f.write("Analysis:\n")
            f.write(f"{result['analysis']}\n\n")

    print('Analysis done! Results are saved in report.md.')