import os
import time
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate


def setup_model(model_id):
    local_root_model_dir = '/mnt/sda/huggingface/data/models'
    model_directory_path = os.path.join(local_root_model_dir, model_id)

    gpu_llm = HuggingFacePipeline.from_model_id(
        model_id=model_directory_path,
        device=0,
        task="text-generation",
    )

    template = """
    You are a friendly chatbot assistant that responds conversationally to users' questions.
    Keep the answers short, unless specifically asked by the user to elaborate on something.
    Question: {question}
    Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    gpu_chain = prompt | gpu_llm.bind(stop=["\n\n"])

    return gpu_chain


def ask_question(llm_chain, question):
    result = llm_chain.batch(question)
    return result


class Timer:
    def __init__(self):
        self._start_time = None

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.perf_counter() - self._start_time
        print(f"Elapsed time: {self.elapsed_time:0.4f} seconds")


if __name__ == '__main__':
    MODEL_ID = "google/flan-t5-xl"
    llm_chain_instance = setup_model(MODEL_ID)
    with Timer():
        response = ask_question(llm_chain_instance, "Describe some famous landmarks in London")
        print(response['question'])
        print("")
        print(response['text'])
