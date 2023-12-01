import openai
import tkinter as tk
from nltk.corpus import wordnet
import re

openai.api_key = 'replace with openai api key'

def get_word_meanings(word):
    synsets = wordnet.synsets(word)

    if not synsets:
        return "Cannot provide meaning for this word, so do not choose a number and just answer the meaning of this word in this sentence yourself"

    meanings = []
    for i, synset in enumerate(synsets):
        meanings.append(f"{i}. {synset.definition()}")

    return '\n'.join(meanings)#extra new line for stop sequences

def remove_number_dot_space(reply):
    pattern = r'^\d+\.\s'
    if re.match(pattern, reply):
        reply = re.sub(pattern, '', reply)
    return reply

def pairs():
    text = text_text.get("1.0", tk.END).strip()
    target_word = word_entry.get()

    meanings = get_word_meanings(target_word)
    prompt_content = "What is the meaning of word \"" + target_word + "\" in the sentence: \"" + text + "\"\n" + "Options" + "\n" + meanings + "\n" + "Repeat the option you selected(include the number):" + "\n"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt_content,
        max_tokens=30,
        temperature=0,
        top_p=0.3,
        frequency_penalty=0,
        presence_penalty=0
    )

    reply = response.choices[0].text.strip()
    reply = remove_number_dot_space(reply)#reply process

    Reply_text.config(state=tk.NORMAL)
    Reply_text.delete(1.0, tk.END)
    Reply_text.insert(tk.END, reply)
    Reply_text.config(state=tk.DISABLED)

window = tk.Tk()
window.title("SenseQuery")
window.geometry("500x250")

word_label = tk.Label(window, text="Target Word")
word_label.pack()
word_entry = tk.Entry(window)
word_entry.pack()

text_label = tk.Label(window, text="Text")
text_label.pack()
text_text = tk.Text(window, height=3)
text_text.pack()

the_button = tk.Button(window, text="Submit", command=pairs)
the_button.pack()

Reply_label = tk.Label(window, text="Reply:")
Reply_label.pack()
Reply_text = tk.Text(window, height=5, state=tk.DISABLED)
Reply_text.pack()

window.mainloop()