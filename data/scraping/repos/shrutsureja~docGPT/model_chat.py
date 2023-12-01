import os
from dataclasses import asdict, dataclass

from dotenv import load_dotenv
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# from utils import format_prompt

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

model_path = os.environ.get("MODEL_PATH")

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.


n_gpu_layers=12,
n_threads=4,
n_batch=1000

# @dataclass
# class GenerationConfig:
#     # sample
#     top_k: int
#     top_p: float
#     temperature: float
#     repetition_penalty: float
#     last_n_tokens: int
#     seed: int

#     # eval
#     batch_size: int
#     threads: int

#     # generate
#     max_new_tokens: int
#     stop: list[str]
#     stream: bool
#     reset: bool


def load_model():
    try:
        # check if the model is already downloaded
        if os.path.exists(model_path):
            print("Loading model...")
            global llm
            # llm = CTransformers(
            #     model=os.path.abspath(model_path),
            #     model_type="mpt",
            #     callbacks=[StreamingStdOutCallbackHandler()],
            # )
            # Make sure the model path is correct for your system!
            llm = LlamaCpp(
                model_path=model_path,
                # temperature=0.75,
                max_tokens=2000,
                n_ctx=512,
                # top_p=1,
                callback_manager=callback_manager, 
                verbose=True, # Verbose is required to pass to the callback manager
            )
            return True
        else:
            raise ValueError(
                "Model not found. Please run `poetry run python download_model.py` to download the model."
            )
    except Exception as e:
        print(str(e))
        raise


if __name__ == "__main__":

    # generation_config = GenerationConfig(
    #     temperature=0.1,
    #     top_k=0,
    #     top_p=0.9,
    #     repetition_penalty=1.0,
    #     max_new_tokens=512,
    #     seed=42,
    #     reset=False,
    #     stream=True,  # streaming per word/token
    #     threads=int(os.cpu_count() / 2),  # adjust for your CPU
    #     stop=["<|im_end|>", "|<"],
    #     last_n_tokens=64,
    #     batch_size=8,
    # )

    # load model if it has already been downloaded. If not prompt the user to download it.
    load_model()

    while True:
        query = input("\nEnter a question: ")
        # query = "Question: " + query
        # if query == "Question: exit":
        if query == "exit":
            break
        if query.strip() == "":
            continue
        try:
            print("Thinking...")
            # call llm with formatted user prompt and generation config
            # response = llm(format_prompt(query), **apredict(generation_config))
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            llm_chain.run(query)
            # response = llm(query)
            # print response
            print("\n")
        except Exception as e:
            print(str(e))
            raise