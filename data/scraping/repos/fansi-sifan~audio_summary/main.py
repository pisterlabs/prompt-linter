import os
import sys
from dotenv import load_dotenv
from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
import whisper

# Load environment variables from .env file
load_dotenv()

# Initialize Anthropic
anthropic = Anthropic()

# Function to transcribe audio
def transcribe_audio(audio_file, model_name="base"):
    model = whisper.load_model(model_name)  # Load the model
    result = model.transcribe(audio_file, fp16=False)  # Transcribe the audio file
    print("Audio to text done")
    print(". ".join(result["text"].split(". ")[:5]))  # Print the first 5 lines of the transcription

    # Store the transcription in a .txt file with the same name as the audio file
    with open(os.path.splitext(audio_file)[0] + "_raw.txt", "w") as f:
        f.write(result["text"])

    return result["text"]

# Function to create completion
def create_completion(prompt_text):
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=100000,
        prompt=prompt_text,
    )
    print(". ".join(completion.completion.split(". ")[:5]))  # Print the first 5 sentences of the completion
    return completion.completion

# Function to summarize conversation
def summarize(split_text):
    prompt_text = f"{HUMAN_PROMPT} This is a conversation transcript between two people. Transcript: {split_text}\n\n First, Split the conversation by spearker, then summarize the conversation by speaker using bullet points. Finally, based on the conversation, write a follow up thank you email to the interviewert{AI_PROMPT}"
    return create_completion(prompt_text)

def read_or_create_file(file_path, creation_func, *args):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            content = f.read()
    else:
        content = creation_func(*args)
        with open(file_path, "w") as f:
            f.write(content)
    return content

def write_results(audio_file):
    raw_file = os.path.splitext(audio_file)[0] + "_raw.txt"
    result = read_or_create_file(raw_file, transcribe_audio, audio_file)

    # Generate Summary based on split
    summary_file = os.path.splitext(audio_file)[0] + "_summary.txt"
    summary = summarize(result)

    with open(summary_file, "a") as f:
        f.write("\nSummary:\n")
        f.write(summary)

# Main function
if __name__ == "__main__":
    audio_file = sys.argv[1]
    write_results(audio_file)
