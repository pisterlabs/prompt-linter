from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
import streamlit as st

import os

# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# # Check API Key
# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.")
#     st.stop()
openai_api_key = st.write(
    "OPENAI_API_KEY",
    os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"],
)

# if "messages" not in st.session_state:
#     st.session_state["messages"] = [{"role": "assistant", "content": "List the products you are interested in:"}]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

def search_for_products(product):
    # Multi-line input for user to list products
    
    ## Pull products from external user

    # If products are given, start the search process
    if product:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
        search_agent = initialize_agent(
            tools=[DuckDuckGoSearchRun(name="Search")],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
        )
        # for product in products:
        # Use the product name as a prompt to the assistant for a search
        prompt = f"Tell me more about the product: {product}"
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run([{"role": "user", "content": prompt}], callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
    return response
