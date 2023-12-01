#!pip install langchain llama-cpp-python
from langchain.llms import OpenAI
from langchain.llms import LlamaCpp

## Cloud
# model = OpenAI()

### Edge
# model = LlamaCpp(model_path="./models/llama-7b-ggml-v2-q4_0.bin", verbose=True, n_threads=8, n_gpu_layers=26)
# model = LlamaCpp(model_path="./models/stable-vicuna-13B-ggml_q4_0.bin", verbose=True, n_threads=8, n_gpu_layers=10)
model = LlamaCpp(model_path="./models/koala-7B.ggml.q4_0.bin", verbose=True, n_threads=8, n_gpu_layers=26)

# Generate text
prompt = "Once upon a time, "

print("Sending prompt: " + prompt)
response = model(prompt)
print(response)