from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.llms import LangChainLLM, HuggingFaceLLM
from llama_index import LangchainEmbedding, ServiceContext
from langchain.llms import HuggingFaceTextGenInference
from llama_index.llms.anyscale import Anyscale
import os


def get_stablelm_context():
    system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
    - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
    - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
    - StableLM will refuse to participate in anything that could harm a human.
    """

    # This will wrap the default prompts that are internal to llama-index
    query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

    # Change default model
    # embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
    hf_predictor = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=20,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        # system_prompt=system_prompt,
        # query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
        model_name="StabilityAI/stablelm-tuned-alpha-3b",
        device_map="cpu",
        # stopping_ids=[50278, 50279, 50277, 1, 0],
        tokenizer_kwargs={"max_length": 4096},
        tokenizer_outputs_to_remove=["token_type_ids"],
        # uncomment this if using CUDA to reduce memory usage
        # model_kwargs={"torch_dtype": torch.float16}
    )
    service_context = ServiceContext.from_defaults(
        chunk_size=1024, llm=hf_predictor, embed_model="local"
    )
    return service_context


def get_falcon_context():
    system_prompt = """# Falcon-7B Instruct
    - You are a helpful AI assistant and provide the answer for the question based on the given context.
    - You answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, you say "I don't know".
    """

    # This will wrap the default prompts that are internal to llama-index
    query_wrapper_prompt = SimpleInputPrompt(">>QUESTION<<{query_str}\n>>ANSWER<<")

    # Change default model
    # embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
    hf_predictor = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=32,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        tokenizer_name="tiiuae/falcon-7b-instruct",
        model_name="tiiuae/falcon-7b-instruct",
        device_map="cpu",
        # stopping_ids=[50278, 50279, 50277, 1, 0],
        tokenizer_kwargs={"max_length": 4096},
        tokenizer_outputs_to_remove=["token_type_ids"],
        # uncomment and add to kwargs if using CUDA to reduce memory usage
        model_kwargs={
            "trust_remote_code": True
        },  # , "load_in_8bit": True} #"torch_dtype": torch.float16
    )
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=hf_predictor,
        embed_model="local",
        query_wrapper_prompt=query_wrapper_prompt,
    )
    return service_context


def get_llama_context():
    # This will wrap the default prompts that are internal to llama-index
    query_wrapper_prompt = SimpleInputPrompt(">>QUESTION<<{query_str}\n>>ANSWER<<")

    # Change default model
    # embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
    hf_predictor = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.6, "do_sample": False},
        tokenizer_name="meta-llama/Llama-2-7b-hf",
        model_name="meta-llama/Llama-2-7b-hf",
        device_map="cpu",
        # stopping_ids=[50278, 50279, 50277, 1, 0],
        # tokenizer_kwargs={"max_length": 4096},
        # uncomment and add to kwargs if using CUDA to reduce memory usage
        model_kwargs={"trust_remote_code": True},  # "torch_dtype": torch.float16
    )
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=hf_predictor,
        embed_model="local",
        query_wrapper_prompt=query_wrapper_prompt,
    )
    return service_context


def get_falcon_tgis_context(temperature, repetition_penalty):
    # system_prompt = """# Falcon-7B Instruct
    # - You are a helpful AI assistant and provide the answer for the question based on the given context.
    # - You answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, you say "I don't know".
    # """

    # print("Creating query_wrapper_prompt")
    # This will wrap the default prompts that are internal to llama-index
    # query_wrapper_prompt = SimpleInputPrompt(">>QUESTION<<{query_str}\n>>ANSWER<<")

    print("Changing default model")
    # Change default model
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

    print(f"Getting server environment variables")
    server_url = os.getenv(
        "TGIS_SERVER_URL", "http://localhost"
    )  # Get server url from env else default
    server_port = os.getenv(
        "TGIS_SERVER_PORT", "8049"
    )  # Get server port from env else default
    print(
        f"Initializing TGIS predictor with server_url: {server_url}, server_port: {server_port}"
    )
    inference_server_url = f"{server_url}:{server_port}/"
    print(f"Inference Service URL: {inference_server_url}")

    tgis_predictor = LangChainLLM(
        llm=HuggingFaceTextGenInference(
            inference_server_url=inference_server_url,
            max_new_tokens=256,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            server_kwargs={},
        ),
    )

    print("Creating service_context")
    service_context = ServiceContext.from_defaults(
        chunk_size=1024, llm=tgis_predictor, embed_model=embed_model
    )
    return service_context


def get_anyscale_context(max_tokens: int = 256):
    # AnyScale Test
    anyscale_llm = Anyscale(
        model="meta-llama/Llama-2-7b-chat-hf",
        api_key="KEY",
        max_tokens=str(max_tokens),
        temperature=0
    )
    service_context = ServiceContext.from_defaults(
        chunk_size=1024, llm=anyscale_llm, embed_model="local:BAAI/bge-small-en-v1.5"
    )
    return service_context
