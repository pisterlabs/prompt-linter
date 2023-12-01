# from langchain.document_loaders import TextLoader
# from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import DirectoryLoader

# loader1 = TextLoader("transcripts/meta.md")
# loader2 = UnstructuredMarkdownLoader("transcripts/meta.md")
loader3 = DirectoryLoader(
    "./",
    glob="transcripts/**/*.*",
    use_multithreading=True,
    max_concurrency=4,
    show_progress=True,
)

# docs1 = loader1.load()
# docs2 = loader2.load()
docs3 = loader3.load()
# print(docs1)
# print(20 * "=")
# print(docs2)
# print(20 * "=")
print(docs3)
