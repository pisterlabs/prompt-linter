import streamlit as st
from streamlit_extras.stateful_button import button
from streamlit_extras.switch_page_button import switch_page
import pandas as pd

from utils.page_control import check_prev_step
from utils.pipeline_control import (
    show_prompts_and_variables,
    show_response_schemas,
    get_prompt,
    parse_md_code,
)
from utils.model_control import get_model

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate

check_prev_step("Prompts", "prompts_done")
check_prev_step("Model", "model_selection_done")

api_config = {
    "api_key": st.session_state["api_key"],
    "api_base": st.session_state["api_base"],
}

strategy = st.session_state.get("strategy")

st.title("Pipeline")

with st.expander("Explore metadata", ):
    metadata = pd.DataFrame(st.session_state.get("metadata"))
    metadata_columns = metadata.columns.tolist()
    st.write("*Metadata*")
    st.dataframe(metadata)
sys_p = st.session_state.get("system_prompt")
rs = st.session_state.get("response_schemas")
code_snippet_p = st.session_state.get("code_snip_prompt")
aggr_sc_p = st.session_state.get("aggr_code_prompt")

with st.expander("Explore prompts", ):
    st.write("#### System prompt")
    matches_sys = show_prompts_and_variables(sys_p)

    st.write(f"#### *User Prompt 1:  {strategy['display_name']}*")
    matches_cs = show_prompts_and_variables(code_snippet_p)

    if len(strategy["prompts"]) > 1:
        st.write(f"#### *User Prompt 2: {strategy['display_name']}*")
        matches_fin = show_prompts_and_variables(aggr_sc_p)
    st.write("#### *Response schemas*")
    rs = show_response_schemas(rs)
    
with st.expander("Play with prompts", expanded=True):
    i = st.slider("Row number", 1, len(metadata), 1) -1
    row = metadata[matches_cs].iloc[i]
    args = dict(zip(matches_cs, row.values))
    st.write("*args*")
    st.write(args)
    formatted_prompt1 = get_prompt(rs, code_snippet_p, args)
    st.write("*formatted prompt1*")
    st.markdown(formatted_prompt1)

if not button("Generate answer", key="first_prompt"):
    st.stop()

with st.spinner("Fetching Answers..."):
    answers = []
    progress_bar = st.progress(0)
    for i in range(len(metadata)):
        row = metadata[matches_cs].iloc[i]
        args = dict(zip(matches_cs, row.values))
        formatted_prompt = get_prompt(rs, code_snippet_p, args)

        llm = get_model(api_conf=api_config)
        output = llm(
            system=sys_p,
            user=formatted_prompt,
            llm=st.session_state["llm_name"],
            temp=st.session_state["temp"]
        )["choices"][0]["message"]["content"]
        answers.append(parse_md_code(output))
        progress_bar.progress((i+1)/len(metadata))

st.write("*answers*")
tabs = st.tabs([*map(str, range(len(answers)))])
for i, ans in enumerate(answers):
    with tabs[i]:
        st.title("Answer {}".format(i+1))
        st.code(ans)

st.session_state.update({"answers":answers})
st.write("Answers saved, navigate to the next step.")
