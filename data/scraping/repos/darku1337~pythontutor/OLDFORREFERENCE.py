import tkinter as tk
from tkinter import simpledialog, scrolledtext
import openai
from pathlib import Path
import tkinter.ttk as ttk

# Set up OpenAI
openai.api_key = "YOUR_OPENAI_API_KEY"

class PythonTutor:

    def __init__(self, master):
        self.master = master
        master.title("Python Tutor")
        
        self.label = tk.Label(master, text="Topic: Introduction to Python")
        self.label.pack(pady=20)
        
        self.chat_box = scrolledtext.ScrolledText(master, width=80, height=20)
        self.chat_box.pack(pady=20)
        self.chat_box.configure(state=tk.DISABLED)
        
        self.user_input = scrolledtext.ScrolledText(master, width=80, height=4)
        self.user_input.pack(pady=20)
        
        self.master.bind("<Return>", self.send_message)

        self.curriculum = [
            "Introduction to Python: Python is a high-level, interpreted programming language...",
            "Python Basics: Variables, Data Types, and I/O...",
            "Control Structures: If statements, loops, and more...",
            # ... [add more curriculum topics as you wish]
        ]
        self.current_topic = self.load_progress()
        self.update_chat("Tutor", self.curriculum[self.current_topic])
        
        self.quiz_question = None

        # Dark mode flag and toggle button
        self.dark_mode = False
        self.toggle_theme_button = tk.Button(master, text="Toggle Dark Mode", command=self.toggle_theme)
        self.toggle_theme_button.pack(pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(master, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)
        self.update_progress()

    def toggle_theme(self):
        if self.dark_mode:
            self.master.configure(bg="white")
            self.label.configure(bg="white", fg="black")
            self.chat_box.configure(bg="white", fg="black")
            self.user_input.configure(bg="white", fg="black")
            self.dark_mode = False
        else:
            self.master.configure(bg="black")
            self.label.configure(bg="black", fg="white")
            self.chat_box.configure(bg="black", fg="white")
            self.user_input.configure(bg="black", fg="white")
            self.dark_mode = True

    def update_progress(self):
        total_topics = len(self.curriculum)
        self.progress["value"] = (self.current_topic / total_topics) * 100
        self.progress.update()

    def send_message(self, event=None):
        message = self.user_input.get("1.0", tk.END).strip()
        self.user_input.delete("1.0", tk.END)
        if message:
            self.update_chat("User", message)

            # If in quiz mode, evaluate the answer
            if self.quiz_question:
                feedback = self.evaluate_answer(self.quiz_question, message)
                self.update_chat("Tutor", feedback)
                self.quiz_question = None
                self.current_topic += 1
                self.save_progress()
                self.update_progress()
                self.next_topic()
            else:
                response = self.ask_gpt(message)
                self.update_chat("Tutor", response)

    def update_chat(self, sender, message):
        self.chat_box.configure(state=tk.NORMAL)
        self.chat_box.insert(tk.END, f"{sender}: {message}\n")
        self.chat_box.configure(state=tk.DISABLED)
        self.chat_box.yview(tk.END)

    def next_topic(self):
        self.label.config(text=f"Topic: {self.curriculum[self.current_topic]}")
        self.update_chat("Tutor", self.curriculum[self.current_topic])
        
        # Prompt for a quiz question based on the topic
        question_prompt = f"Generate a quiz question related to: {self.curriculum[self.current_topic]}"
        response = self.ask_gpt(question_prompt)
        self.quiz_question = response
        self.update_chat("Tutor", self.quiz_question)

    def ask_gpt(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Python tutor."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()

    def evaluate_answer(self, question, answer):
        evaluation_prompt = f"Evaluate the answer to the question '{question}'. The provided answer is: {answer}"
        evaluation = self.ask_gpt(evaluation_prompt)
        return evaluation

    def load_progress(self):
        if Path("progress.txt").exists():
            with open("progress.txt", "r") as f:
                try:
                    return int(f.read().strip())
                except ValueError:
                    return 0
        else:
            return 0

    def save_progress(self):
        with open("progress.txt", "w") as f:
            f.write(str(self.current_topic))

root = tk.Tk()
app = PythonTutor(master=root)
root.mainloop()
