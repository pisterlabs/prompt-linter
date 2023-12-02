from modules.ai.utils.args import get_args
from guidance.models._llama_cpp import LlamaCpp
from llama_cpp import Llama
from langchain.llms.llamacpp import LlamaCpp as LangLlamaCpp

llm: LangLlamaCpp = None
guid: LlamaCpp = None

def create_llm_guidance() -> LlamaCpp:
    """Create instance of LLaMA 2 model for use with guidance"""

    global guid
    if guid != None:
        return guid

    print("Creating llm instance")
    args = get_args()
    llm = LlamaCpp(
        model=args.model_path,
        n_gpu_layers=args.gpu_layers,
        n_batch=512,
        use_mmap=True,
        n_ctx=2048,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        temperature=0,
        top_k=40,
        top_p=1,
        repeat_penalty=1/0.85,
        verbose=False,
    )

    guid = llm

    return llm


def create_llm() -> LangLlamaCpp:
    """Create instance of LLaMA 2 model with LlamaCpp API"""
    global llm

    if llm != None:
        return llm

    args = get_args()
    llm = LangLlamaCpp(
        model_path=args.model_path,
        n_gpu_layers=args.gpu_layers,
        n_batch=512,
        use_mmap=True,
        n_ctx=2048,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        max_tokens=1000,
        temperature=0,
        top_k=40,
        top_p=1,
        repeat_penalty=1/0.85,
        verbose=True,
    )

    return llm

