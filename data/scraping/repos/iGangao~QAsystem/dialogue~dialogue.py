from datetime import datetime
import streamlit as st
from streamlit_chatbox import *
import torch
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass, field

from transformers.generation.utils import GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from vectorstore.vectorstore import vs

TEMPLATE = """已知{question}\n{answer}\n"""
KB = """question: {question}\nanswer: {answer}\n"""
TEMPERATURE = 0.7
HISTORY_LEN = 3
VECTOR_SEARCH_TOP_K = 1
SCORE_THRESHOLD=  0.5


'''
@dataclass
class GeneratingArguments:
    """
    Arguments pertaining to specify the decoding parameters.
    """
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use sampling, use greedy decoding otherwise."}
    )
    temperature: Optional[float] = field(
        default=0.3,
        metadata={"help": "The value used to modulate the next token probabilities."}
    )
    top_p: Optional[float] = field(
        default=0.85,
        metadata={"help": "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."}
    )
    top_k: Optional[int] = field(
        default=5,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."}
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."}
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."}
    )
    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."}
    )
    repetition_penalty: Optional[float] = field(
        default=1.1,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."}
    )
    length_penalty: Optional[float] = field(
        default=2.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation."}
    )

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", None):
            args.pop("max_length", None)
        return args
'''

def load_pretrained() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="baichuan-inc/Baichuan-13B-Chat",
        trust_remote_code=True,
        cache_dir = None,
        revision = "main",
        use_auth_token = None,
        user_fast = False,
        padding_side="right",
    )
    # Load and prepare pretrained models (without valuehead).
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path="baichuan-inc/Baichuan-13B-Chat",
        trust_remote_code=True,
        cache_dir = None,
        revision="main",
        use_auth_token=None,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="baichuan-inc/Baichuan-13B-Chat",
        config=config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir = None,
        revision="main",
        use_auth_token=None,
    )

    # # Register auto class to save the custom code files.
    if hasattr(config, "auto_map") and "AutoConfig" in config.auto_map:
        config.__class__.register_for_auto_class()
    if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:
        tokenizer.__class__.register_for_auto_class()
    if hasattr(config, "auto_map") and "AutoModelForCausalLM" in config.auto_map:
        model.__class__.register_for_auto_class()

    return model, tokenizer

@st.cache_resource
def init_model():

    # generating_args = GeneratingArguments
    model, tokenizer = load_pretrained()
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name="GanymedeNil/text2vec-large-chinese",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    

    model.generation_config = GenerationConfig.from_pretrained(
        "baichuan-inc/Baichuan-13B-Chat",
    )

    if False:
        model.generation_config.max_new_tokens = generating_args.max_new_tokens
        model.generation_config.temperature = generating_args.temperature
        model.generation_config.top_k = generating_args.top_k
        model.generation_config.top_p = generating_args.top_p
        model.generation_config.repetition_penalty = generating_args.repetition_penalty
        model.generation_config.do_sample = generating_args.do_sample
        model.generation_config.num_beams = generating_args.num_beams
        model.generation_config.length_penalty = generating_args.length_penalty
    
    return model, tokenizer, embeddings

chat_box = ChatBox()

def get_messages_history(history_len: int) -> List[Dict]:
    def filter(msg):
        '''
        针对当前简单文本对话，只返回每条消息的第一个element的内容
        '''
        content = [x._content for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        return {
            "role": msg["role"],
            "content": content[0] if content else "",
        }

    # workaround before upgrading streamlit-chatbox.
    def stop(h):
        return False

    history = chat_box.filter_history(history_len=100000, filter=filter, stop=stop)
    user_count = 0
    i = 1
    for i in range(1, len(history) + 1):
        if history[-i]["role"] == "user":
            user_count += 1
            if user_count >= history_len:
                break
    return history[-i:]


def dialogue_page():
    llm, tokenizer, embeddings = init_model()
    vectorstore = vs(embeddings)
    chat_box.init_session()

    with st.sidebar:
        # TODO: 对话模型与会话绑定
        def on_mode_change():
            mode = st.session_state.dialogue_mode
            text = f"已切换到 {mode} 模式。"
            st.toast(text)
            # sac.alert(text, description="descp", type="success", closable=True, banner=True)

        dialogue_mode = st.selectbox("请选择对话模式：",
                                     ["LLM 对话","知识库问答",],
                                     index=1,
                                     on_change=on_mode_change,
                                     key="dialogue_mode",
                                     )

        def on_llm_change():
            model = st.session_state.llm_model
            text = f"已切换到 {model} 模型。"
            st.toast(text)

        llm_model = st.selectbox("选择LLM模型：",
                                ["baichuan-13b","chatglm2-6b"],
                                on_change=on_llm_change,
                                key="llm_model",
                                )
           
        st.session_state["cur_llm_model"] = llm_model
        temperature = st.slider("Temperature：", 0.0, 1.0, TEMPERATURE, 0.05)
        history_len = st.number_input("历史对话轮数：", 0, 10, HISTORY_LEN)


        if dialogue_mode == "知识库问答":
            with st.expander("知识库配置", True):
                kb_top_k = st.number_input("匹配知识条数：", 1, 20, VECTOR_SEARCH_TOP_K)
                score_threshold = st.slider("知识匹配分数阈值：", 0.0, 1.0, float(SCORE_THRESHOLD), 0.01)
                # chunk_content = st.checkbox("关联上下文", False, disabled=True)
                # chunk_size = st.slider("关联长度：", 0, 500, 250, disabled=True)
                
    # Display chat messages from history on app rerun
    chat_box.output_messages()

    # Function for generating BaiChuan response
    def generate_baichuan_response(prompt, history, temperature):
        
        #"top_p":top_p, "max_length":max_length, "repetition_penalty":1
        messages = history
        messages.append({"role": "user", "content": prompt})

        generation_config = llm.generation_config
        generation_config.temperature = temperature
        for response in llm.chat(tokenizer, messages, stream=True, generation_config=generation_config):
            yield response

    def generate_baichuan_prompt(prompt, top_k, score_threshold):
        qas = vectorstore.do_search(prompt, top_k=top_k, score_threshold=score_threshold)
        res = ""
        kb = []
        for qa in qas:
            question = qa["question"]
            answer = qa["answer"]
            res += TEMPLATE.replace("{question}", question).replace("{answer}", answer)
            kb.append(KB.replace("{question}", question).replace("{answer}", answer))
        return res + "请根据以上信息回答问题:\n"+ prompt, kb


    chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter "
    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        history = get_messages_history(history_len)
        chat_box.user_say(prompt)
        
        if dialogue_mode == "LLM 对话":
            chat_box.ai_say("正在思考...")
            text = ""
            for res in generate_baichuan_response(prompt, history, temperature):
                text += res
                chat_box.update_msg(text)
            chat_box.update_msg(text, streaming=False)  # 更新最终的字符串，去除光标

        elif dialogue_mode == "知识库问答":

            chat_box.ai_say([
                f"正在根据知识库生成回答...",
                Markdown("...", in_expander=True, title="知识库匹配结果", state='complete'),
            ])
            text = ""
            prompt, kb = generate_baichuan_prompt(prompt,
                                                top_k=kb_top_k,
                                                score_threshold=score_threshold,)
            
            chat_box.update_msg("\n\n".join(kb), element_index=1, streaming=False)
            for res in generate_baichuan_response(prompt,history, temperature):
                # text += res
                # chat_box.update_msg(text, element_index=0)
                chat_box.update_msg(res, element_index=0, streaming=False)
            
    

    now = datetime.now()

    with st.sidebar:
        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
                "清空对话",
                use_container_width=True,
        ):
            chat_box.reset_history()
            st.experimental_rerun()

    export_btn.download_button(
        "导出记录",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        mime="text/markdown",
        use_container_width=True,
    )
