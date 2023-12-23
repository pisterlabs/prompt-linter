from langchain import ConversationChain
from transformers import AutoTokenizer
import transformers
import torch
import accelerate


from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

PDF_PATH = 'document/red_hat_satellite.pdf'
# have to be huggingface model, suffix with -hf
LLAMA2_MODEL_PATH = '../llama/llama-2-7b-chat-hf'  
#LLAMA2_MODEL_PATH = '../llama/llama-2-13b-chat-hf'  
tokenizer=AutoTokenizer.from_pretrained(LLAMA2_MODEL_PATH)
pipeline=transformers.pipeline(
    "text-generation",
    model=LLAMA2_MODEL_PATH,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    #device="cuda:0",
    device_map="auto",
    max_length=2048,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
    )

sequences = pipeline(
    'Hi! I like cooking. Can you suggest some recipes?\n')
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

def chat():
    LLAMA2=HuggingFacePipeline(
        pipeline=pipeline, 
        model_kwargs={'temperature':0})
    # Prompt
    prompt_template = """
    <s>[INST] <<SYS>>
    {{ You are a helpful AI Assistant}}<<SYS>>
    ###

    Previous Conversation:
    '''
    {history}
    '''

    {{{input}}}[/INST]

    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['input', 'history'])

    chain = ConversationChain(llm=LLAMA2, prompt=prompt)

    result = chain.run("What is the capital Of India?")
    print(result)