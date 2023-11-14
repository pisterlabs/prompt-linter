import streamlit as st
import os
import torch
import transformers
import time
import argparse
from streamlit.logger import get_logger
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
import intel_extension_for_pytorch as ipex
from optimum.intel.generation.modeling import TSModelForCausalLM
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline


logger = get_logger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument("--auth_token",
                    help='HuggingFace authentification token for getting LLaMa2',
                    required=True)

parser.add_argument("--model_id",
                    type=str,
                    choices=["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf"],
                    default="meta-llama/Llama-2-7b-chat-hf",
                    help="Hugging Face model id")

parser.add_argument("--window_len",
                    type=int,
                    help='Chat memory window length',
                    default=5)

parser.add_argument("--dtype",
                    type=str,
                    choices=["float32", "bfloat16"],
                    default="float32",
                    help="bfloat16, float32")

parser.add_argument("--device",
                    type=str,
                    choices=["cpu"],
                    default="cpu",
                    help="cpu")

parser.add_argument("--max_new_tokens",
                    type=int,
                    default=32,
                    help="Max tokens for warmup")

parser.add_argument("--prompt",
                    type=str,
                    default="Once upon time, there was",
                    help="Text prompt for warmup")

parser.add_argument("--num_warmup",
                    type=int, 
                    default=15,
                    help="Number of warmup iterations")

parser.add_argument("--alpha",
                    default="auto",
                    help="Smooth quant parameter")

parser.add_argument("--output_dir",
                    default="./models",
                    help="Output directory for quantized model")

parser.add_argument("--ipex",
                    action="store_true",
                    help="Whether to use IPEX")

parser.add_argument("--jit",
                    action="store_true",
                    help="Whether to enable graph mode with IPEX")

parser.add_argument("--sq",
                    action="store_true",
                    help="Enable inference with smooth quantization")

parser.add_argument("--int4",
                    action="store_true",
                    help="Enable 4 bits quantization with bigdl-llm")


args = parser.parse_args()


if args.ipex:
    try:
        ipex._C.disable_jit_linear_repack()
    except Exception:
        pass
    
if args.jit:
    torch._C._jit_set_texpr_fuser_enabled(False)
    
if args.int4:
    from bigdl.llm.transformers import AutoModelForCausalLM

    
# Check if amp is enabled
amp_enabled = True if args.dtype != "float32" else False
amp_dtype = getattr(torch, args.dtype)


# App title
st.set_page_config(
    page_title="LLaMA2-7b Chatbot",
    page_icon="🦙",
    layout="centered",
)

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today ?"}]
    memory.clear()
    
def get_conversation(llm, window_len=args.window_len):
    # Define memory
    memory = ConversationBufferWindowMemory(k=window_len)
    conversation = ConversationChain(
        llm=llm, 
        verbose=True, 
        memory=memory
    )

    conversation.prompt.template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. You are the AI, so answer all the questions adressed to you respectfully. You generate only when the human asks a question, and don't answer by acting as both a human and AI, remember this!, so don't ever generate text starting with "Human:..". Current conversation:\nAI: How can I help you today ? \n{history}\nHuman: {input}\nAI:"""
    
    return conversation, memory

@st.cache_resource()
def LLMPipeline(temperature, 
                top_p,
                top_k,
                max_length,
                hf_auth,
                repetition_penalty=1.1,
                model_id=args.model_id):
    
    # Initialize tokenizer & model
    tokenizer = LlamaTokenizer.from_pretrained(model_id, token=hf_auth)
    model = LlamaForCausalLM.from_pretrained(model_id,
                                             torch_dtype=amp_dtype,
                                             torchscript=True if args.sq or args.jit else False,
                                             token=hf_auth)
    model = model.to(memory_format=torch.channels_last)
    model.eval()
    
    # Model params
    num_att_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    
    # Apply IPEX llm branch optimizations
    if args.ipex:
        model = ipex._optimize_transformers(model, dtype=amp_dtype, inplace=True)
    
    # Smooth quantization option
    if args.sq:
        model = TSModelForCausalLM.from_pretrained(args.output_dir, file_name="best_model.pt")
    
    # 4bits quantization with bigdl
    if args.int4:
        model.save_pretrained("models/fp32")
        model = AutoModelForCausalLM.from_pretrained("models/fp32", load_in_4bit=True)
    
    # IPEX Graph mode
    if args.jit and args.ipex:
        dummy_input = tokenizer(args.prompt, return_tensors="pt")
        input_ids = dummy_input['input_ids']
        attention_mask = torch.ones(1, input_ids.shape[-1] + 1)
        attention_mask[:, 0] = 0
        past_key_values = tuple(
            [
                (
                    torch.ones(size=[1, num_att_heads, 1, head_dim]),
                    torch.ones(size=[1, num_att_heads, 1, head_dim]),
                )
                for _ in range(num_layers)
            ]
        )
        example_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }
        with torch.no_grad(), torch.autocast(
            device_type=args.device,
            enabled=amp_enabled,
            dtype=amp_dtype if amp_enabled else None,
        ):
            trace_model = torch.jit.trace(model, example_kwarg_inputs=example_inputs, strict=False, check_trace=False)
            trace_model = torch.jit.freeze(trace_model)
            # Use TSModelForCausalLM wrapper since traced models don't have a generate method
            model = TSModelForCausalLM(trace_model, model.config)
        
    # Warmup iterations
    logger.info('[INFO]: Starting warmup.. \n')
    with torch.inference_mode(), torch.no_grad(), torch.autocast(
        device_type=args.device,
        enabled=amp_enabled,
        dtype=amp_dtype if amp_enabled else None
    ):
        for i in range(args.num_warmup):
            start = time.time()
            input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
            output_ids = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=True, top_p=top_p, top_k=top_k)
            gen_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            logger.info('[INFO]: Time generation: %.3f sec' %(time.time()-start))
            logger.info('[INFO]: {}'.format(gen_text))
    logger.info('[INFO]: Warmup finished \n')
    
    # Define HF pipeline
    generate_text = pipeline(model=model,
                             tokenizer=tokenizer,
                             return_full_text=True,
                             task='text-generation',
                             temperature=temperature,
                             top_p=top_p,
                             top_k=top_k,                         
                             max_new_tokens=max_length,
                             repetition_penalty=repetition_penalty)
    
    llm = HuggingFacePipeline(pipeline=generate_text)
    
    # Create langchain conversation
    conversation, memory = get_conversation(llm)
  
    return conversation, memory

# Replicate Credentials
with st.sidebar:
    st.title('🦙 LLaMA2-7b Chatbot')   
    
    # Text generation params
    st.subheader('Text generation parameters')
    temperature = st.sidebar.slider('temperature', min_value=0.1, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    top_k = st.sidebar.slider('top_k', min_value=0, max_value=100, value=20, step=10)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=512, step=8)
    
    # Load conversation
    conversation, memory = LLMPipeline(temperature, top_p, top_k, max_length, args.auth_token)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today ?"}]

# Display chatbot messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🦙" if message["role"] == "assistant" else "🧑‍💻"):
        st.write(message["content"])

# Button to clear chatbot memory
st.sidebar.write('\n')
st.sidebar.write('\n')
_, middle, _ = st.sidebar.columns([.16, 2.5, .1])
with middle :
    clear_button = st.button(':arrows_counterclockwise: Clear Chatbot Memory', on_click=clear_chat_history)


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="🦙"):
        placeholder = st.empty()
        placeholder.markdown("▌")
        response = conversation.predict(input=prompt)
        full_response = ""
        for item in response:
            full_response += item
            placeholder.markdown(full_response + "▌")
            time.sleep(0.04)
        placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
    
    
    
    
    
