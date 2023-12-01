from langchain.llms import Ollama

ollama = Ollama(base_url='http://localhost:11434',
model="metalurgico")

print(ollama("Explique para que serve um protetor auricular."))