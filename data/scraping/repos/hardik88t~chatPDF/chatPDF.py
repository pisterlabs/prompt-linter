import sys
import fitz
import openai

openai.api_key = 'OPENAI_API_KEY'

def get_combined_text(pdf_path):
    doc = fitz.open(pdf_path)
    combined_text = ''
    for page in doc:
        text = page.get_text()
        combined_text += text
    doc.close()
    return combined_text


def ask_question(prompt, combined_text):
    # Truncate or summarize the combined_text to fit within the maximum context length
    max_context_length = 4096
    combined_text = combined_text[:max_context_length]

    messages = [
        {"role": "system", "content": """You would answer three types of questions
        1. Direct Query Questions: These are questions where you would find keywords in text e.g. What is CDE? What was the dataset used in the study?
        2. Indirect Query Questions: These are where no keyword is found e.g. Why was the proposed method used?
        3. Identification of key references that inspire the proposed methodology in the paper"""},
        {"role":"user",'content':combined_text},
        {"role":"assistant","content":"text received now ask anything about it."},
        {"role":"user","content":prompt}
    ]

    # Check if the total token count exceeds the maximum allowed
    total_tokens = sum(len(message["content"].split()) for message in messages)
    if total_tokens > 800:
        print("=== The conversation exceeds the maximum token limit.===")
        return

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, max_tokens=800, temperature=0.2
    )

    reply = chat.choices[0].message.content
    print(reply)

def main():
    # Use when Running from Colab/Notebook
    pdf_path = input("Enter the path to the PDF file: ")
    combined_text = get_combined_text(pdf_path)

    # Use when running from Command Line
    # pdf_path = sys.argv[1]
    # combined_text = get_combined_text(pdf_path)

    while True:
        prompt = input("Enter your question (or 'quit' to exit): ")
        if prompt.lower() == "quit":
            break
        ask_question(prompt, combined_text)

if __name__ == '__main__':
    main()