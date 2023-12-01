#Generative Lecture Summarizer
#Shahariar Ali Bijoy, Tamás Lukács

import os
import sys
import openai
import jinja2
import pdfkit
import tiktoken
import whisper
from pprint import pprint
import tkinter
import customtkinter as ctk
from tkinter import filedialog
from serpapi import GoogleSearch
import subprocess



openai.api_key = ""
serpapi_key = ""
model_name = "gpt-3.5-turbo"
model_max_tokens = 4096


def transcribe(media_path):
    model = whisper.load_model("base")
    transcript = model.transcribe(media_path)
    return transcript["text"]

def get_summary(transcript, summary_length):
    summary_length = int(summary_length) * 3  # chatbot words are about 3 times longer than normal words?
    system_prompt = f"You are a student, who creates notes about the most important information and overarching themes of the provided university lecture. The resulting notes should contain bulletpoints separated by '-' symbols"
    print("System Prompt: \n" + system_prompt)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Universtiy Lecture: \n" + transcript}
        ]
    )
    return completion.choices[0].message.content

def extract_topics(summary):
    system_prompt = "Find the the 3 most important scientific areas to search for from the following summary. Separate the terms with ',' symbols. The terms should be maxium 4 words long."

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summary}
        ]
    )
    return completion.choices[0].message.content

def get_related_articles(topic):
    params = {
    "api_key": serpapi_key,
    "engine": "google_scholar",
    "q": topic.replace(".", ""),
    "hl": "en",
    "num": "3"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    
    return results["organic_results"]

def count_tokens(text):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)


def generate_pdf(title, summary, topics, papers):
    # create html list
    summary = summary.replace("-", "</li><li>")

    authors = []
    for paper in papers:
        try:
            authors.append(paper["publication_info"]["authors"][0]["name"])
        except:
            authors.append("No author found")

    pdf_topics = ["" for i in range(3)]
    for i in range(3):
        try:
            pdf_topics[i] = topics.split(',')[i].replace(".", "")
        except:
            pass

    pdf_papers = ["" for i in range(9)]
    for i in range(9):
        try:
            pdf_papers[i] = papers[i]["title"] + "<br> Author: " + authors[i] + "<br> Link: " + papers[i]["link"] + "<br>"
        except:
            pass


    context = {
        'title': title,
        'summary': summary,
        't1': pdf_topics[0],
        't2': pdf_topics[1],
        't3': pdf_topics[2],
        'p1': pdf_papers[0],
        'p2': pdf_papers[1],
        'p3': pdf_papers[2],
        'p4': pdf_papers[3],
        'p5': pdf_papers[4],
        'p6': pdf_papers[5],
        'p7': pdf_papers[6],
        'p8': pdf_papers[7],
        'p9': pdf_papers[8]
    }
    template_loader = jinja2.FileSystemLoader('./')
    template_env = jinja2.Environment(loader=template_loader)
    html_template = 'summary-template.html'
    template = template_env.get_template(html_template)
    output_text = template.render(context)
    config = pdfkit.configuration(wkhtmltopdf='/usr/local/bin/wkhtmltopdf')
    output_path = os.path.join(f"summary {title}.pdf")
    pdfkit.from_string(output_text, output_path, configuration=config)
    subprocess.call(['open', output_path])

def create_summary(file_path, title):
    if file_path.endswith(".txt"):
        with open(file_path, 'r') as file:
            transcript = file.read().replace('\n', '')
    else:
        print("Transcribing...")
        transcript = transcribe(file_path)
        with open(os.path.join(f"transcript-{title}.txt"), 'w') as file:
            file.write(transcript)

    token_count = count_tokens(transcript)
    if token_count > model_max_tokens:
        print(f"The provided lecture is {token_count/model_max_tokens} times longer the current model allows. Please provide a shorter lecture or change to a different model. Model relevant model limitations: GPT-3.5: up to about 25 minues of audio, GPT-4: about 3 hours of audio")
        sys.exit(1)

    print("Summarizing...")
    summary = get_summary(transcript, summary_length=100)
    print("\n\n\n################ Summary: ################\n\n" + summary)
    print("\n\n\nExtracting topics...")
    topics = extract_topics(summary)
    print("\n\n\n################ Topics: ################\n\n", topics)
    print("\n\n\nExtracting related articles...")
    related_papers = []
    for topic in topics.split(','):
        for paper in get_related_articles(topic):
            related_papers.append(paper)
    print("\n\n\n################ Related Articles: ################\n\n")
    pprint(related_papers)
    print("\n\n\n################ Generating PDF... ################\n\n")
    generate_pdf(title, summary, topics, related_papers)
    print("\n\n\n################ All Done! \n\n\n################")


def upload_audio_file():
    file_path = filedialog.askopenfilename(title="Select a file", filetypes=(("mp3 files","*.mp3"),("wav files","*.wav"), ("all files","*.*")))
    if file_path:
        input_box.delete(0, tkinter.END)
        input_box.insert(0, file_path)

def main():
    root = tkinter.Tk()
    root.geometry("600x200")
    root.title("Lectutre Summarizer")

    # Create title input box
    title_label = ctk.CTkLabel(master=root, text="Lecture Title: ", font=("Arial", 15))
    title_label.place(x=20, y=20)
    title_input_box = ctk.CTkEntry(master=root, width=430)
    title_input_box.place(x=150, y=20)

    # Create file input box
    global input_box
    input_box = ctk.CTkEntry(master=root, width=380)
    input_box.place(x=200, y=70)
    upload_button = ctk.CTkButton(master=root, text="Select Lecture File", corner_radius=10, command=upload_audio_file)
    upload_button.place(x=20, y=70)

    # Create generate summary button
    generate_button = ctk.CTkButton(master=root, text="Generate Summary", corner_radius=10, command=lambda: create_summary(input_box.get(), title_input_box.get()))
    generate_button.place(x=230, y=150)
    root.mainloop()


if __name__ == "__main__":
    main()
