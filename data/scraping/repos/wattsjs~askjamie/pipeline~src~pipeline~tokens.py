import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

tokenizer_name = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokenizer = tiktoken.get_encoding(tokenizer_name.name)


# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""],
)
