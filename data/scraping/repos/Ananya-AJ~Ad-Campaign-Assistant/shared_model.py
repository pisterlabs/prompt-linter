from langchain.llms import GPT4All

def load_model():
    # Add your model loading logic here
    local_path = "Models/llama-2-7b-32k-instruct.Q4_0.gguf"
    model = GPT4All(model=local_path, backend="gptj", max_tokens=1024, n_predict=256)
    return model

# Load the model when this module is imported
