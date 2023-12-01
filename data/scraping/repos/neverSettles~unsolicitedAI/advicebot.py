import openai

# Initialize a list to store past advice and tasks
past_advice_and_tasks = []

def get_past_advice_and_tasks_window():
    # Retrieve the last few entries
    return past_advice_and_tasks[-5:]

def add_advice_and_task(advice_and_task):
    # Add new advice and task to the list
    past_advice_and_tasks.append(advice_and_task)

def get_most_recent_advice_and_task():
    # Get the most recent advice and task
    if past_advice_and_tasks:
        return past_advice_and_tasks[-1]
    else:
        return "No recent advice and tasks available."

def transcribe_audio(file_path):
    # Transcribe an audio file using Whisper
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
    return transcript["text"]

def get_advice_and_task(question):
    # Enhanced prompt for actionable advice and a task
    prompt = (
        "Based on the following user conversation, provide a concise and actionable piece of advice. "
        "Then, suggest a specific, practical task that the user can perform. Format the response as a JSON object "
        "with two keys: 'advice' for the advice, and 'task' for the suggested task. User conversation: '{}'"
    ).format(question)

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        temperature=0.7
    )

    # Process and add the advice and task to the list
    response_text = response.choices[0].text.strip()
    
    # Check if the response text contains a newline character
    if '\n' in response_text:
        advice, task = response_text.split('\n', 1)
    else:
        advice = response_text
        task = "No specific task provided."

    # Splitting the response into advice and task
    parts = response_text.split("\n")
    advice = parts[0].replace("Advice: ", "").strip() if len(parts) > 0 else "No advice provided."
    task = parts[1].replace("Task: ", "").strip() if len(parts) > 1 else "No specific task provided."

    advice_and_task = {"Advice": advice, "Task": task}
    add_advice_and_task(advice_and_task)

    return advice_and_task

def read_text_file(file_path):
    # Read text from a text file
    with open(file_path, "r") as file:
        return file.read()

# Example usage
openai.api_key = 'APIKEY'

# Transcribe an audio file and get advice based on the transcription
audio_file= open("file.mp3", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
# print(transcript.text)
transcript = read_text_file("hackathonconvo.txt")

advice_and_task = get_advice_and_task(transcript)
print(advice_and_task["Advice"])
# print("What I can help you do:", advice_and_task["Task"])

# Get and print past advice and task window
print("Past advice and tasks window:", get_past_advice_and_tasks_window())

# Get and print the most recent advice and task
print("Most recent advice and task:", get_most_recent_advice_and_task())
