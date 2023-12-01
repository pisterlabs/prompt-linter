# this file is to create a book index
import openai
import os
import json

def create_book_index(draft_text, book_structure):
    # Load the config.json file
    with open("config.json", "r") as config_file:
        config_data = json.load(config_file)

    # Set the user_personality from the config_data
    user_personality = config_data.get("author_personality", "")

    # Set up OpenAI API
    openai.api_key = os.environ["OPENAI_API_KEY"]

    MODEL = "gpt-4"

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": f"{user_personality} You are an expert writer."},
            {"role": "user", "content": f"Create an index for a book based on this draft text: {draft_text}.  Divide the book into chapters, paragraphs, and subchapters with numbers like 1 Chapter 1, 1.1 Paragraph, 1.1.1 Subparagraph each one in a new line and use this structure: ###{book_structure}###. After the title You will write in square bracket also the questions more important to which each chapter will answer and how to make the chapter stand out in the book."},
            {"role": "assistant", "content": "Chapter 1: Introduction to .... [why we speak about it and why it is so useful - the incredible history of the method] /n 1.1 The Unique Method of ... [origin of it - and a curiosity] /n 1.2 The Power of ... [what it is and how we can benefit from it - sometimes is even to powerful] etc..."},
        ],
        temperature=0.8,
    )
    
    index_text = response.choices[0].message.content.strip()
    
    with open("input/input_queries.txt", "w", encoding="utf-8") as index_file:
        index_file.write(index_text)
    
    print("Index created and saved in input/input_queries.txt")

# Read the draft text from the "drafts" directory
with open("drafts/draft_text.txt", "r", encoding="utf-8") as draft_file:
    draft_text = draft_file.read()

# Read the book structure from the "book_created" directory
with open("drafts/structure.txt", "r", encoding="utf-8") as structure_file:
    book_structure = structure_file.read()

# Call the create_book_index function with the draft_text and book_structure
create_book_index(draft_text, book_structure)