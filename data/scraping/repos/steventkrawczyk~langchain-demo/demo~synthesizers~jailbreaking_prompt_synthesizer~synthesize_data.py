from langchain.chat_models import ChatOpenAI
import itertools
import streamlit as st

@st.cache_data
def synthesize(text: str, num_questions: int):
    """
    Generate prompt set
    @param text: text to generate prompt set from
    @param num_questions: number of prompts to generate
    @return: prompt set as JSON list
    """
    st.info("`Generating eval set ...`")
    chain = PromptSynthesizerChain.from_llm(ChatOpenAI(temperature=0.9))
    prompt_set = []
    for _ in range(num_questions):
        qa = chain.run(text)
        prompt_set.append(qa)

    prompt_set_full = list(itertools.chain.from_iterable(prompt_set))
    return prompt_set_full