import os
import openai
import subprocess
import time
from tkinter import messagebox

output_folder = "output"
model = "gpt-3.5-turbo"

def generate_flashcards(input_text, language, input_language, loading_label, button, progress_bar, output_folder, model):
    # GPT Prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Create anki flashcards for the following {language} words in {input_language}, only one output per word is allowed.: {input_text}. The output format must be: word in spanish;phrase in spanish;reading in spanish;word in english;translated phrase. Example: hola;hola, que tal;hola;hello;hello how are you doing"}
    ]

    loading_label.config(text="Generating flashcards...")

    # GPT settings
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=2000
    )

    generated_flashcards = response["choices"][0]["message"]["content"]

    timestamp = time.strftime("%Y%m%d%H%M%S")
    file_name = f"flashcards_{timestamp}_{language}.txt"

    file_path = os.path.join(output_folder, file_name)

    try:
        # Save the generated flashcards to a file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(generated_flashcards)

        # Update the loading label
        loading_label.config(text="Flashcards generated successfully!")

        # Show success message
        messagebox.showinfo("Flashcard Generator", "Flashcards generated successfully!")

        # Open the output folder
        folder_path = os.path.abspath(output_folder)
        subprocess.Popen(["explorer", folder_path])
    except Exception as e:
        # Update the loading label
        loading_label.config(text="Error generating flashcards!")

        # Show error message
        messagebox.showerror("Error", f"Error generating flashcards: {str(e)}")

    # Enable the generate button
    button.config(state="normal")

    # Reset the progress bar
    progress_bar.stop()
    progress_bar["value"] = 0