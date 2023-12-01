import openai


openai.api_key = os.getenv('OPENAI_API_KEY')


model_engine = "text-davinci-002"

# Extract text from the PDF using a PDF extraction tool
pdf_text = text

batch_size = 1024

# Split the PDF text into chunks of the desired size
text_chunks = [pdf_text[i:i+batch_size] for i in range(0, len(pdf_text), batch_size)]

# Initialize an empty list to store the generated text
generated_text = []
for chunk in text_chunks:
    prompt = "Write a Markdown version of the PDF using the following text, keep the text intact and no suggestions should be added:\n\n" + chunk
    completion = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=len(chunk), n=1,stop=None,temperature=0)
    generated_text.append(completion.choices[0].text)

# Combine the generated text into a single string
full_text = "".join(generated_text)

# Split the text into paragraphs
paragraphs = full_text.split("\n\n")

# Insert the images into the appropriate position in the list of paragraphs
for i, paragraph in enumerate(paragraphs):
    if i == 2:  # Insert image after the third paragraph
        paragraphs.insert(i+1, "![Alt text](" + lst[0] + ")")

# Combine the paragraphs into a single string
formatted_text = "\n\n".join(paragraphs)

# Use Markdown syntax to format the message
formatted_message = "# PDF Text\n\n" + formatted_text

# Save the message to a file
with open("pdf_text.md", "w") as f:
    f.write(formatted_message)
