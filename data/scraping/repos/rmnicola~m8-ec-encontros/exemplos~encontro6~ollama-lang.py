from langchain.llms import Ollama
ollama = Ollama(model="mistral")
print(ollama("why is the sky blue. respond with at most 5 words"))
