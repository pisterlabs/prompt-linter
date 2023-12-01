import os
from anthropic import Anthropic

api_key = os.getenv("ANTHROPIC_API_KEY")

anthropic = Anthropic(api_key=api_key)

# Define the exercise prompt
exercise_prompt = """
Human: Please generate a JavaScript challenge that covers a fundamental concept for a junior JavaScript developer. The challenge should be in the form of comments (//), and it should not include code that would result in errors. Make it a code completion exercise where the student needs to complete the code.
Assistant:
"""



# Use Anthropic to generate the JavaScript exercise code based on the prompt
response = anthropic.completions.create(
    model="claude-2",
    prompt=exercise_prompt,
    max_tokens_to_sample=150
)

# Extract the generated code from the response
exercise_code = response.completion

# Save the exercise code to a JavaScript file
exercise_file_path = "exercise.js"
with open(exercise_file_path, "w") as exercise_file:
    exercise_file.write(exercise_code)

print(f"Exercise file '{exercise_file_path}' generated.")
