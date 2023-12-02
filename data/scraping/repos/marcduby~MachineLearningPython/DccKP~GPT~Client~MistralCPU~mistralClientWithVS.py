

# imports
from langchain.llms import CTransformers

# constants 
FILE_MODEL_MISTRAL_7B = "/scratch/Javaprog/Data/ML/Models/slimopenorca-mistral-7b.Q8_0.gguf"

# methods


# main
if __name__ == "__main__":
    llm = CTransformers(
        model = FILE_MODEL_MISTRAL_7B,
        config = {
            'max_new_tokens' : 128,
            'temperature': 0.01
        },
        streaming = True
    )




