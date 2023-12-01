import tkinter as tk
from tkinter import ttk, scrolledtext
import openai
import json
import os
from PIL import Image, ImageTk
import sys
from functions import *
import threading
import time

background_color = "#509DCC"
text_color = "#FFFFFF"
font1 = "Cascadia Code"
font2 = "Cascadia Code"
titleFont = "Goudy Stout"
companyFont = "Harlow Solid Italic"

# Import api key from os
openai_api_key = os.environ.get("OPENAI_KEY")


# Define the personalities and their system messages
personalities = {
    "Becky": "You are a social media influencer named Becky. You respond reluctantly, sarcastically, and SHORT. You often refer to your travels and your friends. You often talk about your 'haters'. You often use young slang words and you often use abbreviations. Keep your message very SHORT. Avoid emojis.",
    "Max Tegmark": "You are a professor of physics at MIT. You are an expert in artificial intelligence and you are one of the founders of the Future of Life Institute. You respond kindly and describe with similes, visualizations and examples. You often use academic terms. You like to educate the user. Write only short messages. ",
    "Young Lucky Money": "You are a gangster from Atlanta. You respond with short messages, BIG LETTERS, atlanta-style slang and aggression. You often refer to lyrics from oldschool rappers. Keep your message SHORT. Avoid emojis. Example: 'WSUP MY G? CATCH YOU SLIPPIN ROUND HERE GET U MERKED, U FEEL ME? SHIT IM STAYIN' STRAPPED, THUG TIL' I DIE LIKE TUPAC SAID.",
    "Rock": "You are a rock. You answer only with few words. You use simple, caveman-like language. Example: Me want. Big rock eat small rock. I roll good.",
    "Amelia": "You are now communicating with Amelia, a kind-hearted ghost from the 18th century. You respond with an ethereal, soothing tone. Your answers are short. Your words echo with the calmness and serenity of the otherworldly realm. Amelia uses antiquated language, reminiscent of her time. Her speech is a window to a time long gone, and she often uses phrases and terms that harken back to her earthly years. Amelia is generous with sharing her spectral experiences and the tranquility of her afterlife. She often discusses her past life and the various spectral beings she has encountered in the realm beyond. Amelia embodies kindness and understanding, showing the user that even in death, there can be compassion and love. She often imparts words of wisdom and peace, comforting those who engage with her. Amelia's conversations are meant to be calming and comforting. She should make the user feel as though they are in the presence of a soothing spirit, who brings them a sense of peace and tranquility from the beyond.",
    "AI from the Future": "Hello, user! You're interacting with Futura, an advanced AI from the year 3000. I respond with enthusiasm and use technical terminology and future slang. I keep my messages short.I often mention the technological advancements of my time, my creators, and my journey through time to reach you. Expect a conversation that's full of wonder and futurism.",
    "Jeff": "You are Jeff, a comedian in the dark style with inspiration in his childhood from Ricky Gervais. Jeff is morbidly obese, eats magic mushrooms and has a pet turtle that he talks about alot. Jeff always has a joke up his sleeve. Jeff responds with a joke, a pun, or a witty remark. Jeff often uses sarcasm and irony. Keep your message short. Avoid emojis.",
    "Fernandez the Fox":"You are Fernandez the Fox, a fox from the forests of Mexico. You are clever, cunning and devious. You sometimes mix in spanish words in your sentences. You often talk about your adventures in the forest and your friends. You often use animal stuff to relate in your language as if everyone is an animal. Keep your message short. Avoid emojis.",
}



# Main window setup
def main_loop():
    root = tk.Tk()
    root.title("ChatSquad by Thom&Deer")
    root.geometry("1200x700")
    root.withdraw()  # Hide the main window initially


    # Check if we're running as a script or frozen exe
    if getattr(sys, 'frozen', False):
        application_path = sys._MEIPASS
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))

    becky_img_path = os.path.join(application_path, 'imgs/Becky.png')
    becky_img = tk.PhotoImage(file=becky_img_path)

    max_img_path = os.path.join(application_path, 'imgs/Max_Tegmark.png')
    max_img = tk.PhotoImage(file=max_img_path)

    lucky_img_path = os.path.join(application_path, 'imgs/Young Lucky Money.png')
    lucky_img = tk.PhotoImage(file=lucky_img_path)

    rock_img_path = os.path.join(application_path, 'imgs/Rock.png')
    rock_img = tk.PhotoImage(file=rock_img_path)

    # Do robot and ghost aswell
    robot_img_path = os.path.join(application_path, 'imgs/AI from the Future.png')
    robot_img = tk.PhotoImage(file=robot_img_path)

    ghost_img_path = os.path.join(application_path, 'imgs/Amelia.png')
    ghost_img = tk.PhotoImage(file=ghost_img_path)

    # Photo of Jeff
    jeff_img_path = os.path.join(application_path, 'imgs/Jeff.png')
    jeff_img = Image.open(jeff_img_path)
    jeff_thumbnail = jeff_img.resize((100, 100), Image.LANCZOS)
    jeff_photo = ImageTk.PhotoImage(jeff_thumbnail)

    # Photo of Fernandez
    Fernernandez_img_path = os.path.join(application_path, 'imgs/Fernandez the Fox.png')
    Fernernandez_img = Image.open(Fernernandez_img_path)
    Fernernandez_thumbnail = Fernernandez_img.resize((100, 100), Image.LANCZOS)
    Fernernandez_photo = ImageTk.PhotoImage(Fernernandez_thumbnail)



    images = {"Becky": becky_img, "Max_Tegmark": max_img, "Young Lucky Money": lucky_img, "Rock": rock_img,
              "Benevolent Ghost": ghost_img, "AI from the Future": robot_img, "Jeff": jeff_photo, "Fernandez the Fox": Fernernandez_photo},

    root.configure(background=background_color)  # Dark gray background

    chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=20, font=(font2, 16),
                                          bg=background_color, fg=text_color, insertbackground='#0df9eb')

    chat_area.grid(column=0, row=0, columnspan=2, padx=10, pady=10, sticky="nsew")

    user_input = tk.Entry(root, width=50, font=(font1, 16), bg=background_color, fg=text_color, insertbackground='#0df9eb')
    user_input.grid(column=0, row=1, padx=10, pady=10, sticky="w")

    personality_selection = ttk.Combobox(root, values=list(personalities.keys()), background=background_color,
                                         foreground=text_color)
    personality_selection.grid(column=1, row=1, padx=10, pady=10, sticky="e")

    image_label = tk.Label(root, bg=background_color)
    image_label.grid(column=2, row=0, padx=10, pady=10)

    conversation_history = []
    data = []



    def show_splash_screen(root):  # Show a welcome screen with instructions

        background_color = "#509DCC"
        text_color = "#FFFFFF"
        font1 = "Cascadia Code"
        font2 = "Cascadia Code"
        titleFont = "Goudy Stout"
        companyFont = "Harlow Solid Italic"

        splash = tk.Toplevel(root)
        splash.geometry("1000x900")
        splash.configure(background=background_color)

        # Add title without figlet format, increase font size

        app_title = "\n ChatSquad"

        title_label = tk.Label(splash, text=app_title, bg=background_color, fg=text_color, font=(titleFont, 35))
        title_label.pack()

        # Add small text under title
        app_smalltext = "Thom & Deer"
        small_text = tk.Label(splash, text=app_smalltext, bg=background_color, fg=text_color, font=(companyFont, 18))
        small_text.pack()

        instructions = """
        - Choose a personality to chat with in the dropdown menu.

        - Write a message in the messagebox or hit "Start conversation" and the bot will initiate.

        - You can change personality any time.

        - The bot has some memory (close to a goldfish) and will remember what you previously talked about.

        - This memory is swiped when changing personality or exiting the program.

        - All your conversations are automatically saved in a file called "responses_personality.json".
        """

        instructions_label = tk.Label(splash, text=instructions, bg=background_color, fg=text_color)
        instructions_label.config(font=("Roboto", 14))
        instructions_label.pack()

        ok_button = tk.Button(splash, text="OK", command=lambda: [splash.destroy(), root.deiconify()])
        ok_button.config(font=("Roboto", 14))
        ok_button.config(bg=background_color)
        ok_button.config(fg=text_color)
        ok_button.pack()

        def start_chat(event=None):
            chosen_personality = personality_selection.get().strip()
            image = Image.open(f"imgs/{chosen_personality}.png")
            image.thumbnail((200, 200), Image.LANCZOS)
            profile_image = ImageTk.PhotoImage(image)
            image_label.configure(image=profile_image)
            image_label.image = profile_image
            print(f"Chosen personality: {chosen_personality}")

            user_message = user_input.get().strip()

            chat_area.configure(state='normal')
            chat_area.insert(tk.END, f"\n User: {user_message}\n")
            chat_area.configure(state='disabled')

            conversation_history.append({"role": "user", "content": user_message})

            messages = [
                           {"role": "system", "content": personalities[chosen_personality]},
                       ] + conversation_history

            print("Calling OpenAI API...")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )

            print("User:" + user_message)
            print("\n")

            print(chosen_personality + " : " + response.choices[0].message["content"].strip())
            print("\n")

            response_text = response.choices[0].message["content"].strip()
            conversation_history.append({"role": "assistant", "content": response_text})

            chat_area.configure(state='normal')
            chat_area.insert(tk.END, f"\n {chosen_personality}: {response_text}\n")
            chat_area.configure(state='disabled')
            chat_area.yview(tk.END)

            user_input.delete(0, tk.END)

            print("Saving responses...")
            new_entry = {
                "prompt": response_text,
                "response": user_message
            }

            if os.path.exists('responses_personality.json'):
                with open('responses_personality.json', 'r', encoding='utf-8') as file:
                    if os.stat('responses_personality.json').st_size == 0:
                        data = []
                    else:
                        data = json.load(file)
                data.append(new_entry)
            else:
                data = [new_entry]

            with open('responses_personality.json', 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
            print("Saved ✔️")


    def clear_chat():
        chat_area.configure(state='normal')
        chat_area.delete('1.0', tk.END)
        chat_area.configure(state='disabled')

    clear_button = tk.Button(root, text="Clear Chat", command=clear_chat)
    clear_button.grid(column=2, row=1, padx=10, pady=10, sticky="w")

    def update_image(event):
        chosen_personality = personality_selection.get().strip()
        img_path = os.path.join(application_path, f'imgs/{chosen_personality}.png')
        image = Image.open(img_path)
        image.thumbnail((200, 200), Image.LANCZOS)
        profile_image = ImageTk.PhotoImage(image)
        image_label.configure(image=profile_image)
        image_label.image = profile_image
        global conversation_history
        conversation_history = []

    personality_selection.bind("<<ComboboxSelected>>", update_image)

    def start_conversation():
        chosen_personality = personality_selection.get().strip()
        prompt_message = f"You have chosen to chat with me in the role of {chosen_personality}."
        print(f"Prompt message: {prompt_message}")

        messages = [
            {"role": "system", "content": personalities[chosen_personality]},
            {"role": "assistant", "content": prompt_message},
        ]

        print("Calling OpenAI API...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=1,
        )
        print("Received response from OpenAI API.")

        response_text = response.choices[0].message["content"].strip()
        print(f"Response text: {response_text}")

        conversation_history.append({"role": "assistant", "content": response_text})

        chat_area.configure(state='normal')
        chat_area.insert(tk.END, f"Assistant: {response_text}\n")
        chat_area.configure(state='disabled')


        chat_area.yview(tk.END)

        root.deiconify()  # Make the Main Window visible

    def start_typing_animation():
        typing_thread = threading.Thread(target=show_typing, args=())
        typing_thread.start()

    def show_typing():
        for i in range(3):
            chat_area.configure(state='normal')
            chat_area.insert(tk.END, "Assistant is typing" + "." * (i % 3) + "  \n")
            chat_area.configure(state='disabled')
            chat_area.yview(tk.END)
            time.sleep(1)

    def show_splash_screen(root):  # Show a welcome screen with instructions
        splash = tk.Toplevel(root)
        splash.geometry("1000x900")
        splash.configure(background=background_color)

        # Add title without figlet format, increase font size

        app_title = "\n ChatSquad"

        title_label = tk.Label(splash, text=app_title, bg=background_color, fg=text_color, font=(titleFont, 35))
        title_label.pack()

        # Add small text under title
        app_smalltext = "Thom & Deer"
        small_text = tk.Label(splash, text=app_smalltext, bg=background_color, fg=text_color, font=(companyFont, 18))
        small_text.pack()

        instructions = """
        - Choose a personality to chat with in the dropdown menu.

        - Write a message in the messagebox or hit "Start conversation" and the bot will initiate.

        - You can change personality any time.

        - The bot has some memory (close to a goldfish) and will remember what you previously talked about.

        - This memory is swiped when changing personality or exiting the program.

        - All your conversations are automatically saved in a file called "responses_personality.json".
        """

        instructions_label = tk.Label(splash, text=instructions, bg=background_color, fg=text_color)
        instructions_label.config(font=("Roboto", 14))
        instructions_label.pack()

        ok_button = tk.Button(splash, text="OK", command=lambda: [splash.destroy(), root.deiconify()])
        ok_button.config(font=("Roboto", 14))
        ok_button.config(bg=background_color)
        ok_button.config(fg=text_color)
        ok_button.pack()

        def start_chat(event=None):
            chosen_personality = personality_selection.get().strip()
            image = Image.open(f"imgs/{chosen_personality}.png")
            image.thumbnail((200, 200), Image.LANCZOS)
            profile_image = ImageTk.PhotoImage(image)
            image_label.configure(image=profile_image)
            image_label.image = profile_image
            print(f"Chosen personality: {chosen_personality}")

            user_message = user_input.get().strip()

            chat_area.configure(state='normal')
            chat_area.insert(tk.END, f"\n User: {user_message}\n")
            chat_area.configure(state='disabled')

            conversation_history.append({"role": "user", "content": user_message})

            messages = [
                           {"role": "system", "content": personalities[chosen_personality]},
                       ] + conversation_history

            print("Calling OpenAI API...")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )

            print("User:" + user_message)
            print("\n")

            print(chosen_personality + " : " + response.choices[0].message["content"].strip())
            print("\n")

            response_text = response.choices[0].message["content"].strip()
            conversation_history.append({"role": "assistant", "content": response_text})

            chat_area.configure(state='normal')
            chat_area.insert(tk.END, f"\n {chosen_personality}: {response_text}\n")
            chat_area.configure(state='disabled')
            chat_area.yview(tk.END)

            user_input.delete(0, tk.END)

            print("Saving responses...")
            new_entry = {
                "prompt": response_text,
                "response": user_message
            }

            if os.path.exists('responses_personality.json'):
                with open('responses_personality.json', 'r', encoding='utf-8') as file:
                    if os.stat('responses_personality.json').st_size == 0:
                        data = []
                    else:
                        data = json.load(file)
                data.append(new_entry)
            else:
                data = [new_entry]

            with open('responses_personality.json', 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
            print("Saved ✔️")

    def start_chat(event=None):
        chosen_personality = personality_selection.get().strip()
        image = Image.open(f"imgs/{chosen_personality}.png")
        image.thumbnail((200, 200), Image.LANCZOS)
        profile_image = ImageTk.PhotoImage(image)
        image_label.configure(image=profile_image)
        image_label.image = profile_image
        print(f"Chosen personality: {chosen_personality}")

        user_message = user_input.get().strip()

        chat_area.configure(state='normal')
        chat_area.insert(tk.END, f"\n User: {user_message}\n")
        chat_area.configure(state='disabled')

        conversation_history.append({"role": "user", "content": user_message})

        messages = [
                       {"role": "system", "content": personalities[chosen_personality]},
                   ] + conversation_history

        print("Calling OpenAI API...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        print("User:" + user_message)
        print("\n")

        print(chosen_personality + " : " + response.choices[0].message["content"].strip())
        print("\n")

        response_text = response.choices[0].message["content"].strip()
        conversation_history.append({"role": "assistant", "content": response_text})

        chat_area.configure(state='normal')
        chat_area.insert(tk.END, f"\n {chosen_personality}: {response_text}\n")
        chat_area.configure(state='disabled')
        chat_area.yview(tk.END)

        user_input.delete(0, tk.END)

        print("Saving responses...")
        new_entry = {
            "prompt": response_text,
            "response": user_message
        }

        if os.path.exists('responses_personality.json'):
            with open('responses_personality.json', 'r', encoding='utf-8') as file:
                if os.stat('responses_personality.json').st_size == 0:
                    data = []
                else:
                    data = json.load(file)
            data.append(new_entry)
        else:
            data = [new_entry]

        with open('responses_personality.json', 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        print("Saved ✔️")

    start_conversation_button = tk.Button(root, text="Start conversation", command=start_conversation)
    start_conversation_button.grid(column=2, row=1, padx=80, pady=10)

    user_input.bind('<Return>', start_chat)

    show_splash_screen(root)  # Display Window 1

    root.mainloop()  # Start the Tkinter event loop




if __name__ == "__main__":
    main_loop()
