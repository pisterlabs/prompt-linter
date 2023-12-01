import os
import tkinter as tk
from tkinter import messagebox

import openai
import requests
from PIL import Image, ImageTk

from main import generate_image, moderate

openai.api_key = os.getenv("OPENAI_API_KEY")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OpenAI Python")
        self.geometry("400x200")
        self.choice = tk.StringVar(value='image')

        label = tk.Label(self, text="Choose an option:", font=("Arial", 14))
        label.pack(pady=10)

        image_button = tk.Radiobutton(self, text="Image Generation", variable=self.choice, value="image",
                                      font=("Arial", 12))
        image_button.pack()

        text_button = tk.Radiobutton(self, text="Text Moderation", variable=self.choice, value="text",
                                     font=("Arial", 12))
        text_button.pack()

        self.text_entry = tk.Entry(self, width=40, font=("Arial", 12))
        self.text_entry.pack(pady=10)

        submit_button = tk.Button(self, text="Submit", command=self.process_choice,
                                  font=("Arial", 12))
        submit_button.pack()

    def process_choice(self):
        choice = self.choice.get()
        user_input = self.text_entry.get()

        if choice == "image":
            url = None
            try:
                url = generate_image(user_input)
            except openai.error.InvalidRequestError as err:
                messagebox.showerror("Error", str(err))
            if url:
                self.show_image(url)
            else:
                messagebox.showerror("Error", "Image generation failed!")

        elif choice == "text":
            output = moderate(user_input)
            self.display_output(output)

    def show_image(self, url):
        response = requests.get(url, stream=True)
        img = Image.open(response.raw)
        img = img.resize((512, 512), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        image_window = tk.Toplevel(self)
        image_window.title("Generated Image")
        panel = tk.Label(image_window, image=img)
        panel.image = img
        panel.pack()

    def display_output(self, output):
        result_window = tk.Toplevel(self)
        result_window.title("Text Moderation Result")

        result_label = tk.Label(result_window, text="Result:", font=("Arial", 18, "bold"))
        result_label.grid(row=0, column=0, columnspan=3, pady=10)

        row_number = 1
        categories_title_label = tk.Label(result_window, text="Categories", font=("Arial", 16, "bold"))
        categories_title_label.grid(row=row_number, column=0, padx=10, pady=5, sticky="w")

        scores_title_label = tk.Label(result_window, text="Scores", font=("Arial", 16, "bold"))
        scores_title_label.grid(row=row_number, column=2, padx=10, pady=5, sticky="w")

        row_number += 1
        for key, value in output["categories"].items():
            category_label = tk.Label(result_window, text=key, font=("Arial", 14))
            category_label.grid(row=row_number, column=0, padx=10, pady=5, sticky="w")

            category_value_label = tk.Label(result_window, text=str(value), font=("Arial", 14))
            category_value_label.grid(row=row_number, column=1, padx=10, pady=5, sticky="w")
            if value:
                category_value_label.config(fg="red")
            else:
                category_value_label.config(fg="green")

            if key in output["category_scores"]:
                score_value_label = tk.Label(result_window, text=f'{output["category_scores"][key]:.6f}',
                                             font=("Arial", 14))
                score_value_label.grid(row=row_number, column=2, padx=10, pady=5, sticky="w")

            row_number += 1


if __name__ == "__main__":
    app = App()
    app.mainloop()
