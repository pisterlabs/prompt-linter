from langchain.text_splitter import RecursiveCharacterTextSplitter


def vectorize_article(article_path):
    with open(article_path) as f:
        text = f.read()
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents([text])
    print(texts[0])
    print(texts[1])
    return texts
