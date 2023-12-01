import json
import openai

def extract_content(data):
    # Extract the content from the JSON
    return data['content']

def generate_flashcard(chunk, multiple=False, retries=3):
    # Generate a flashcard for a chunk of text
    for _ in range(retries):
        # Generate the flashcard
        if multiple:
            prompt = f"Given the following text, create multiple flashcards with a question and answer for each. Text: {chunk}"
        else:
            prompt = f"Summarize the following text into a question and answer for a flashcard: {chunk}"
        
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=200,
        )

        # Extract the flashcards from the response
        flashcard_texts = response.choices[0].text.strip().split('Flashcard ')
        flashcards = []
        for flashcard_text in flashcard_texts[1:]:  # Skip the first item, which is empty
            if 'Question: ' in flashcard_text and 'Answer: ' in flashcard_text:
                question, answer = flashcard_text.split('Answer: ')
                question = question.replace('Question: ', '').strip()
                flashcards.append((question, answer.strip()))
        
        if flashcards:
            return flashcards

    # If we've tried the maximum number of retries and still haven't gotten a valid flashcard, return None
    return None

def generate_flashcards_from_json(json_data, multiple=False):
    flashcards = []
    for data in json_data:  # iterate over each dictionary in the list
        content = extract_content(data)
        flashcard = generate_flashcard(content, multiple)
        if flashcard is not None:
            flashcards.append(flashcard)
    return flashcards

# Test the functions
# import os
# os.chdir("C:/Users/Matt/Documents/Projects/demand-forecasting/_helpers")
# load json from ./output/level_1_volume_1.json
with open('./output/test.json') as f:
    s_json = json.load(f)

flashcards = generate_flashcards_from_json(s_json, multiple=False)
for question, answer in flashcards:
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print()
