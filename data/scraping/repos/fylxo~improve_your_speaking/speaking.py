import openai
import tkinter as tk
from tkinter import filedialog, Menu, Toplevel
from tkinter.scrolledtext import ScrolledText  # Import ScrolledText

# Set the OpenAI API key
openai.api_key = '.....'

# Function to open the audio file and start the operation
def open_and_transcribe():
    file_path = filedialog.askopenfilename(filetypes=[("MP3 Files", "*.mp3")])
    if file_path:
        transcript_audio(file_path)

# Function to transcribe the audio and get the response
def transcript_audio(file_path):
    try:
        audio_file = open(file_path, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        text = transcript['text']

        # Make an analysis with OpenAI
        evaluation_criteria = """
As an artificial intelligence, my role is to evaluate candidates' language skills based on the standards set by the IELTS exam. I will provide a detailed assessment for each of the four evaluation areas, associating the corresponding band score with each area.

Fluency and Coherence: Evaluation: (Band X)
Lexical Resource: Evaluation: (Band X)
Grammatical Range and Accuracy: Evaluation: (Band X)
Pronunciation: Evaluation: (Band X)

I will also be able to explain the errors in detail and offer specific suggestions on how to improve in each evaluation area.
Errors Made: (Describe errors)
Correct Alternatives: (Provide examples of how to improve )
"""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": evaluation_criteria},
                {"role": "user", "content": text},
            ]
        )
        content = response["choices"][0]["message"]["content"]

        # Open a separate window for the model output
        output_window = Toplevel(root)
        output_window.title("Output del Modello")
        output_window.resizable(True, True)  # Make the output window resizable

        output_text = ScrolledText(output_window, wrap='none')
        output_text.pack(expand=True, fill='both')  # Make the text box expandable
        output_text.insert(tk.END, content)

    except Exception as e:
        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, f"Errore: {str(e)}")

# Create main window
root = tk.Tk()
root.title("Audio Transcription with OpenAI")

# Create menu bar
menu_bar = Menu(root)
root.config(menu=menu_bar)

# File menu
file_menu = Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Apri", command=open_and_transcribe)
file_menu.add_separator()
file_menu.add_command(label="Esci", command=root.quit)

root.mainloop()