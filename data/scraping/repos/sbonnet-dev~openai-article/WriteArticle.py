import os
import openai
import re
import time

from md2pdf.core import md2pdf

OPENAI_KEY = "ENTER_YOUR_OPENAI_KEY"

def callOpenAI(text, max_tokens=150, temperature=0.3):
    # Set OpenAI API key
    openai.api_key = OPENAI_KEY
    model = "text-davinci-003"
    prompt = text

    # Call OpenAI API for text completion
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text.strip()

def generate_outline(thema, language):
    question = f"Create an outline for an essay with short terms about '{thema}' in {language}"
    response = callOpenAI(question)
    return re.sub(r'\n+', '\n', response).strip()

def generate_paragraph(chapter, previous_chapter, thema, language):
    paragraph_type = "an introduction" if chapter == 1 else "a paragraph"
    question = f"Write in {language} {paragraph_type} about '{chapter}' and make a transition with the previous paragraph: '{previous_chapter}' in a book context concerning '{thema}'"
    return callOpenAI(question, max_tokens=4000, temperature=0.9)

############################################
################### MAIN ################### 
############################################

book = ""
os.system("clear")

# Prompt user for language and thematic topic
language = input("Language: (french/english): ")
thema = input("Enter the article thematic: ")

resp = "n"
while resp == "n":
    # Generate outline
    outline = generate_outline(thema, language)
    print(outline + "\n")

    # Prompt user to confirm using the generated table of contents
    resp = input("Do you want to write an article with this table of contents (y/n)? ")

summary = outline.splitlines()
book += f"#{thema}\n"

print("\nWriting document ...")
for chapter, previous_chapter in zip(summary, summary[:-1]):
    print("Writing paragraph =>", chapter)
    
    if len(book) > 0 and (chapter[0] == "I" or chapter[0] == "V"):
        book += f"\n###{chapter}\n"
    else:
        book += f"\n####{chapter}\n"

    paragraph = generate_paragraph(chapter, previous_chapter, thema, language)
    book += f"{paragraph}\n\n"

# Convert the book to PDF using md2pdf
md2pdf("./article.pdf", book, None, None, None)
# Uncomment the line below to open the generated PDF file
# os.system("open ./article.pdf")
