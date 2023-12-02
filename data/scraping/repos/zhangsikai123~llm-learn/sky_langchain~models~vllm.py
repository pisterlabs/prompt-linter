from langchain.llms.vllm import VLLM


llm = VLLM(
    model="/nfs_beijing/sikai/weight/llama/TheBloke/",
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=128,
    top_k=10,
    top_p=0.95,
    temperature=0.8,
)
