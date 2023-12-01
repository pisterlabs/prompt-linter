from langchain.text_splitter import RecursiveCharacterTextSplitter

# This is a long document we can split up.
with open("README.md") as f:
    pg_work = f.read()

print(f"You have {len([pg_work])} document")

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=150,
    chunk_overlap=20,
)

texts = text_splitter.create_documents([pg_work])

print(f"You have {len(texts)} documents")

print("Preview:")
print(texts[0].page_content, "\n")
print(texts[1].page_content)
