from langchain.llms import CTransformers, HuggingFacePipeline # to use CPU only
from langchain.chat_models import ChatOpenAI
import os
import openai
from dotenv import load_dotenv
from torch import cuda, bfloat16
import transformers


def load_llm_ctra_llama27b():
    """
    loading CTransformers model
    """
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.0
    )
    return llm


def load_llm_ctra_llama2_13b():
    """
    Req: https://huggingface.co/GrazittiInteractive/llama-2-13b/blob/main/README.md
    """
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "GrazittiInteractive/llama-2-13b",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.0
    )
    return llm



def load_llm_gpt35():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        openai_api_key=openai.api_key, 
        model="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=1500,
        )
    return llm

def load_llm_gpt4():
    load_dotenv()
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY_GPT4"), 
        model="gpt-4",
        temperature=0.0,
        max_tokens=3000,
        )
    return llm


def load_llm_tokenizer_llama2_13b_hf():
    """
    Ref: https://github.com/pinecone-io/examples/blob/master/learn/generation/llm-field-guide/llama-2/llama-2-13b-retrievalqa.ipynb

    """
    load_dotenv()
    model_id = 'meta-llama/Llama-2-13b-chat-hf'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16,
        llm_int8_enable_fp32_cpu_offload=True
    )

    # begin initializing HF items, need auth token for these
    hf_auth = os.getenv('HF_AUTH_TOKEN')
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    device_map = {
        "transformer.word_embeddings": "0",
        "transformer.word_embeddings_layernorm": "0",
        "lm_head": "cpu",
        "transformer.h": "0",
        "transformer.ln_f": "0",
        "model.embed_tokens.weight": "cpu",
        "model.layers.0.input_layernorm.weight": "cpu",
        "model.layers.0.mlp.down_proj.weight": "cpu",
        "model.layers.0.mlp.gate_proj.weight": "cpu", # TODO: still a lot model layers to be set. we should use device map auto
    }

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=hf_auth,
    )
    model.eval()

    # initialize llama2 13B tokenizer
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )
    
    # huggingface transformers pipeline as our llm

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )
    llm = HuggingFacePipeline(pipeline=generate_text)

    return llm
