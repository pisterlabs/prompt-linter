from langchain.text_splitter import SpacyTextSplitter

with open("/Users/tamasmakos/dev/langflow/text_/pdfs/Output/pdf_text.txt") as f:
    text = f.read()

text_splitter = SpacyTextSplitter(chunk_size=2000, pipeline='xx_sent_ud_sm')
tokenized_texts = text_splitter.split_text(text)

for tokenized_text in tokenized_texts:
    # Save the tokenized text to a file
    with open("/Users/tamasmakos/dev/langflow/text_/pdfs/Output/tokenized_text.txt", "a") as f:
        f.write(tokenized_text)

dataset = 
