import os
import tkinter as tk
import ast
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def display_colors(colors):
    root = tk.Tk()

    for color in colors:
        frame = tk.Frame(root, bg=color, height=150, width=200)
        frame.pack(fill='both')

    root.mainloop()


def generate_palette(prompt_text):
    prompt = f"""
    You are a color palette generating assistant that responds to text prompts for color palettes.
    Based on the following prompt: '{prompt_text}', please generate a palette of 5 colors in the format: ["#color1", "#color2", "#color3", "#color4", "#color5"]
    """

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0.3
    )

    # Error checking for AI response
    try:
        colors = ast.literal_eval(response.choices[0].text.strip())
        if isinstance(colors, list) and all(isinstance(item, str) for item in colors):
            display_colors(colors)
        else:
            print(f"Unexpected response: {response.choices[0].text.strip()}")
    except (ValueError, SyntaxError):
        print(f"Couldn't parse response: {response.choices[0].text.strip()}")


generate_palette("A beautiful sunset")
