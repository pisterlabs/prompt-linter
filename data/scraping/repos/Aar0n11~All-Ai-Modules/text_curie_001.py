
import tkinter as tk
import openai
import os

file_path = os.path.join(os.path.dirname(__file__), "Api key.txt")

with open(file_path, "r") as f:
    api_key = f.read().strip()

openai.api_key = api_key;

def generate_response(prompt, length=1024, context=None):
    if context is None:
        context = []

    response = openai.Completion.create(
        engine="text-curie-001",
        prompt="\n".join(context) + "\n" + prompt,
        max_tokens=length,
        n=1,
        stop=None,
        temperature=0.7,
    ).choices[0].text
    return response.strip()

class ChatBotApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title("ChatBot")
        self.geometry("400x500")

        self.history = tk.Text(self, wrap=tk.WORD, bg="#f0f0f0")
        self.history.pack(fill=tk.BOTH, expand=True)

        self.input = tk.Entry(self)
        self.input.pack(fill=tk.X, padx=10, pady=10)
        self.input.bind("<Return>", self.on_enter)

        self.context = []

    def on_enter(self, event):
        user_input = self.input.get()
        self.history.insert(tk.END, "You: " + user_input + "\n")
        self.input.delete(0, tk.END)
        self.history.see(tk.END)

        if user_input.lower() in ["bye", "quit", "exit"]:
            response = generate_response("Goodbye!", context=self.context)
            self.history.insert(tk.END, "Chatbot: " + response + "\n")
            self.destroy()
        else:
            self.context.append(user_input)
            response = generate_response(user_input, context=self.context)
            self.history.insert(tk.END, "Chatbot: " + response + "\n")

if __name__ == '__main__':
    app = ChatBotApp()
    app.mainloop()

