import tkinter as tk
from tkinter import simpledialog, scrolledtext
import openai
from pathlib import Path
import tkinter.ttk as ttk
import threading

# Set up OpenAI
openai.api_key = "key"

class PythonTutor:
    def __init__(self, master):
        self.master = master
        self.dark_mode = False
        master.title("Cheryl - Your Python Tutor")
        master.geometry("1000x800")

        self.label = tk.Label(master, text="Cheryl - Your Python Tutor", font=('Arial', 24))
        self.label.pack(pady=20)

        self.chat_box = scrolledtext.ScrolledText(master, width=120, height=30, font=('Arial', 14))
        
        self.chat_box.tag_config("assistant_color", foreground="dark blue")
        self.chat_box.tag_config("user_color", foreground="dark green")
        self.chat_box.pack(pady=20)
        self.chat_box.configure(state=tk.DISABLED)

        self.user_input = scrolledtext.ScrolledText(master, width=120, height=4, font=('Arial', 14))
        self.user_input.pack(pady=20)
        self.master.bind("<Return>", self.send_message)

        self.toggle_button = tk.Button(master, text="Toggle Dark Mode", command=self.toggle_dark_mode)
        self.toggle_button.pack(pady=20)

        self.help_button = tk.Button(master, text="Help", command=self.show_help)
        self.help_button.place(relx=0.95, rely=0.05, anchor=tk.NE)

        # Initialize a StringVar for the label text and create the label
        self.thinking_label = tk.Label(master, text="", font=('Arial', 14))
        self.thinking_label.pack(pady=20)

        self.curriculum = [
            "Introduction to Python - history and printing with Python!",
            "Python Basics: Variables, Data Types, and I/O",
            "Control Structures: If statements, loops, and more",
            "Python Functions: Definition, Calling, and Scope",
            "Python Lists, Dictionaries, and Data Structures",
            "Python OOP: Classes, Objects, and Inheritance",
            "File Handling in Python",
            "Python Modules and Libraries",
            "Error Handling: Try, Except, Finally",
            "Advanced Topics: List Comprehensions, Lambda Functions",
            "Python and Databases: SQLite, MySQL",
            "Web Scraping with Python: Beautiful Soup, Requests",
            "Web Development with Flask",
            "Python for Data Science: NumPy, Pandas",
            "Visualization with Python: Matplotlib, Seaborn"
        ]

        self.current_topic = self.load_progress()
        self.teaching_stage = 0
        self.correct_answers_count = 0  # Track correct answers for the current topic
        self.conversation_history = []  # New attribute to store conversation history
        self.quiz_mode = False  # New attribute to indicate if the tutor is in quiz mode

        self.teach_topic()

    def toggle_dark_mode(self):
        if self.dark_mode:
            self.master.config(bg='white')
            self.label.config(bg='white', fg='black')
            self.chat_box.config(bg='white', fg='black')
            self.user_input.config(bg='white', fg='black')
            self.toggle_button.config(bg='white', fg='black')
            self.help_button.config(bg='white', fg='black')
            self.dark_mode = False
        else:
            self.master.config(bg='black')
            self.label.config(bg='black', fg='white')
            self.chat_box.config(bg='black', fg='white')
            self.user_input.config(bg='black', fg='white')
            self.toggle_button.config(bg='black', fg='white')
            self.help_button.config(bg='black', fg='white')
            self.dark_mode = True

    def show_help(self):
        help_text = "This is a Python Tutor. It will teach you the following topics:\n\n"
        help_text += "\n".join(self.curriculum)
        help_text += "\n\nYou can interact with the AI by typing in the input box and pressing Enter. "
        help_text += "At any point, you can ask the AI to generate a quiz about the current topic by typing 'generate quiz'. "
        help_text += "After the quiz question is presented, you can type your answer in the input box. The AI will then evaluate your answer."
        help_window = tk.Toplevel(self.master)
        help_window.title("Help")
        help_label = tk.Label(help_window, text=help_text)
        help_label.pack()

    def teach_topic(self):
        if self.teaching_stage == 0:
            system_message = "You are a Python tutor assistant. Your role is to teach Python to a complete beginner with no coding experience. Remember to provide the teaching directly and do not refer to online tutorials or resources."
            self.update_chat("system", system_message)
            intro = f"Hello! Are you ready to start learning about {self.curriculum[self.current_topic]}?"
            self.update_chat("assistant", intro)
        elif self.teaching_stage == 1:
            intro_response = self.ask_gpt(f"Give a brief introduction to {self.curriculum[self.current_topic]}")
            self.update_chat("assistant", intro_response)
            self.update_chat("assistant", "Did you understand the introduction? If so, we can move on to more details. If not, please ask your questions.")
        elif self.teaching_stage == 2:
            detailed_response = self.ask_gpt(f"Give a detailed explanation of {self.curriculum[self.current_topic]}")
            self.update_chat("assistant", detailed_response)
            self.update_chat("assistant", "Did you understand the details? If so, we can move on to the quiz. If not, please ask your questions.")
        elif self.teaching_stage == 3:
            self.update_chat("assistant", "Are you ready for a quiz on this topic?")
            self.quiz_mode = True
            self.correct_answers_count = 0  # Reset the counter when a new topic starts

    def send_message(self, event=None):
        message = self.user_input.get("1.0", tk.END).strip()
        self.user_input.delete("1.0", tk.END)
        if message:
            self.update_chat("user", message)
            # Update the "Thinking..." label immediately
            self.thinking_label.config(text="Thinking...")
            ai_thread = threading.Thread(target=self.handle_message, args=(message,))
            ai_thread.start()

    def handle_message(self, message):
        if self.quiz_mode:
            # The user is answering a quiz question
            self.quiz_mode = False
            # Evaluate the user's answer
            feedback = self.ask_gpt(f"How did the user do on the quiz question: {message}")
            self.update_chat("assistant", feedback)
            self.teaching_stage = 0
            self.correct_answers_count = 0  # Track correct answers for the current topic
            self.current_topic += 1
            if self.current_topic < len(self.curriculum):
                self.teach_topic()
            else:
                self.update_chat("assistant", "Congratulations! You've completed the entire curriculum.")
        elif message.lower() == "generate quiz":
            # The user has requested a quiz, so generate a quiz question
            question = self.ask_gpt(f"Generate a quiz question about {self.curriculum[self.current_topic]}")
            self.update_chat("assistant", question)
            self.quiz_mode = True
            self.correct_answers_count = 0  # Reset the counter when a new topic starts
        elif message.lower() in ["yes", "continue"]:
            if not self.quiz_mode or (self.quiz_mode and self.correct_answers_count >= 3):
                self.teaching_stage += 1
                self.teach_topic()
        
        elif self.quiz_mode and self.correct_answers_count < 3:  # If not yet answered 3 questions correctly for the topic
            if message.lower() == self.current_answer.lower():
                self.correct_answers_count += 1
                if self.correct_answers_count < 3:
                    # Ask another question related to the current topic
                    question = self.get_quiz_question(self.current_topic)
                    self.update_chat("assistant", question)
                else:
                    self.update_chat("assistant", "Well done! You've answered 3 questions correctly. Would you like to move on to the next topic?")
                    self.quiz_mode = False
            else:
                self.update_chat("assistant", "That's not correct. Try again or type 'skip' to move to the next question.")
        elif message.lower() == "no": 
            self.update_chat("assistant", "Please ask any questions you have about the topic.")
        else:
            response = self.ask_gpt(message)
            self.update_chat("assistant", response)
        # Clear the "Thinking..." label after the response has been sent
        self.thinking_label.config(text="")

    def update_chat(self, sender, message):
        if sender != "system":
            self.chat_box.configure(state=tk.NORMAL)
            self.chat_box.insert(tk.END, f"{'Cheryl' if sender == 'assistant' else 'User'}: {message}\n", "assistant_color" if sender == "assistant" else "user_color")
            self.chat_box.insert(tk.END, "\n")
            self.chat_box.configure(state=tk.DISABLED)
            self.chat_box.yview(tk.END)
        # Add the message to the conversation history
        self.conversation_history.append({"role": sender, "content": message})

    def ask_gpt(self, prompt):
        # Include the conversation history in the messages for the GPT-3 API
        messages = self.conversation_history + [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message['content'].strip()

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

    def get_quiz_question(self, topic_index):
        # Prompt GPT-3 to generate a quiz question for the topic
        topic = self.curriculum[topic_index]
        prompt = f"Generate a quiz question about {topic}."
        response = self.ask_gpt(prompt)
        self.current_answer = self.ask_gpt(f"What's the answer to the question: '{response}' related to {topic}?")  # Capture the answer for later verification
        return response

    def get_quiz_answer(self, topic_index):
        return self.current_answer


root = tk.Tk()
app = PythonTutor(master=root)
root.mainloop()
