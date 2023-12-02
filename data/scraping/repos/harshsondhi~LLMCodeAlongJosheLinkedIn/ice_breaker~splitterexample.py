from langchain.text_splitter import CharacterTextSplitter

with open("some_data/FDR_State_of_Union_1944.txt") as file:
    speech_text = file.read()

print(len(speech_text))
print(len(speech_text.split()))

# text_spliter = CharacterTextSplitter(separator="\n\n", chunk_size=1000)
# texts = text_spliter.create_documents([speech_text])
# print(texts[0])

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
texts = text_splitter.split_text(speech_text)
print(len(texts))
