import textract
import os
import openai
import tiktoken


# Extract the raw text from each PDF using textract
text = (
    textract.process(
        "data/fia_f1_power_unit_financial_regulations_issue_1_-_2022-08-16.pdf",
        method="pdfminer",
    )
    .decode("utf-8")
    .replace("  ", " ")
    .replace("\n", "; ")
    .replace(";", " ")
)


# Example prompt -
document = "<document>"
template_prompt = f"""Extract key pieces of information from this regulation document.
If a particular piece of information is not present, output "Not specified".
When you extract a key piece of information, include the closest page number.
Use the following format:
0. Who is the author
1. What is the amount of the "Power Unit Cost Cap" in USD, GBP and EUR
2. What is the value of External Manufacturing Costs in USD
3. What is the Capital Expenditure Limit in USD

Document: \"\"\"{document}\"\"\"
0. Who is the author: Tom Anderson (Page 1)
1."""


# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j


def extract_chunk(document, template_prompt):

    prompt = template_prompt.replace("<document>", document)

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return "1." + response["choices"][0]["text"]


# Initialise tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

results = []

chunks = create_chunks(text, 1000, tokenizer)
text_chunks = [tokenizer.decode(chunk) for chunk in chunks]

for chunk in text_chunks:
    print(results[-1])


groups = [r.split("\n") for r in results]

# zip the groups together
zipped = list(zip(*groups))
zipped = [x for y in zipped for x in y if "Not specified" not in x and "__" not in x]


# Example prompt -
template_prompt = f"""Extract key pieces of information from this regulation document.
If a particular piece of information is not present, output "Not specified".
When you extract a key piece of information, include the closest page number.
Use the following format:
0. Who is the author
1. How is a Minor Overspend Breach calculated
2. How is a Major Overspend Breach calculated
3. Which years do these financial regulations apply to

Document: \"\"\"{document}\"\"\"
0. Who is the author: Tom Anderson (Page 1)
1."""
print(template_prompt)


results = []

for chunk in text_chunks:
    groups = [r.split("\n") for r in results]


# zip the groups together
zipped = list(zip(*groups))
zipped = [x for y in zipped for x in y if "Not specified" not in x and "__" not in x]
