import os
from openai_utils import generate_prompts

# TODO
# Pivot back towards using this as a local automation script.
# Create function in main.py that iterates though all the books, creates the prompts, sends api calls to ChatGPT for translation, and stitches translations together



def main():
    """Generates literal and natural sounding translation prompts for GPT-4 using ChatGPT Plus."""

    
    BOOK_NUM = 1
    TEXT_FILE = f'/Users/josh/Local Documents/Novels/Memorize/Raws/메모라이즈-{BOOK_NUM}권.txt'
    OUTPUT_DIR = f'/Users/josh/Local Documents/Novels/Memorize/GPT-4 Input Prompts/book-{BOOK_NUM}/'
    MODEL = 'gpt-4'
    MAX_TOKENS = 2700

    # Create output dir if not exists already
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(TEXT_FILE, 'r') as f:
        text = f.read()
        literal_translation_prompts = generate_prompts(
            "Translate the following text to english.\nText:\n", text, MODEL, MAX_TOKENS)
        natural_translation_prompts = generate_prompts(
            "Translate this to english. If necessary, make small adjustments to the translation to make it sound more natural.:\n", text, MODEL, MAX_TOKENS)

    for i in range(len(literal_translation_prompts)):
        with open(OUTPUT_DIR + f"literal-{i}.txt", "w") as f:
            f.write(literal_translation_prompts[i])
    
    for i in range(len(natural_translation_prompts)):
        with open(OUTPUT_DIR + f"natural-{i}.txt", "w") as f:
            f.write(natural_translation_prompts[i])
    
    


if __name__ == '__main__':
    main()
