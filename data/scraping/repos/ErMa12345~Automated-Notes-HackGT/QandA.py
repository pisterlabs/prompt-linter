import openai

# Set your OpenAI API key here
api_key = "API_KEY"

file_path = "Transcriptions/TranscribedAudio.txt"
output_file_path = "summary.txt"  # Path to save the summary

# Read the lecture text from a text file
try:
    with open(file_path, "r") as file:
        lecture_text = file.read().strip()
except FileNotFoundError:
    print("Error: Lecture text file not found.")
    exit()
except Exception as e:
    print(f"An error occurred: {str(e)}")
    exit()

# Check if lecture text is empty
if not lecture_text:
    print("Error: Lecture text is empty.")
    exit()


# Make a request to ChatGPT-3.5 API for summarization
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that generates 10 question and answer pairs"},
        {"role": "user", "content": lecture_text}
    ],
    api_key=api_key
)

try:
    summary = response.choices[0].message["content"].strip()
except KeyError:
    print("Error: Unexpected API response format.")
    exit()

# Print the summarized lecture
print("Summarized Lecture:")
print(summary)

# Write the summary to a text file
try:
    with open(output_file_path, "w") as summary_file:
        summary_file.write(summary)
    print(f"Summary written to {output_file_path}")
except Exception as e:
    print(f"An error occurred while writing the summary: {str(e)}")