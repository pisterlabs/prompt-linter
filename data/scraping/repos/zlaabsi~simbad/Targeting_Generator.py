import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

st.sidebar.success("Select an agent above.")
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

#st.title("🔎 Competition Screener")

#st.set_page_config(page_title="Competition Screener", page_icon="🔎")

st.markdown("# Targeting Generator")
st.sidebar.header("Targeting Generator")
st.write(
    """Generate Microsegments and Apply Specs previously stored. Enjoy!"""
)



if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Generate Microsegments and Apply Specs previously stored."}
    ]

# ... Votre code précédent ...

if target_product := st.chat_input(placeholder="Write a target product, for example : fitness app"):
    st.session_state.messages.append({"role": "user", "content": target_product})
    st.chat_message("user").write(target_product)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Remplacez les placeholders {target_product} dans 'goal' et 'instruction'
    goal = ["Generate Microsegments for a " + target_product + " and apply Specs previously stored"]
    instruction = [
        "Act as a Marketing expert, generate a list of 10 Micro-Segments to grow userbase of a " + target_product,
        "Use the supplied input file SIMBAD_SPEC_LP_V1.txt as a structure for generating a landing page for each of the Microsegments. Write the results to a single file for each of the Micro-Wegments. REMEMBER to format your response as JSON, using double quotes (\"\") around keys and string values, and commas (,) to separate items in arrays and objects. IMPORTANTLY, to use a JSON object as a string in another JSON object, you need to escape the double quotes.",
        "Use the supplied input file SIMBAD_SPEC_AD_V1.txt as a structure for generating an Ad for each of the Microsegments.  Write the results to a single file for each of the Micro-Wegments. REMEMBER to format your response as JSON, using double quotes (\"\") around keys and string values, and commas (,) to separate items in arrays and objects. IMPORTANTLY, to use a JSON object as a string in another JSON object, you need to escape the double quotes."
    ]

    constraints = [
        "If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.",
        "Ensure the tool and args are as per current plan and reasoning"]

    # Define your prompt template here for Targeting Generator
    targeting_generator_prompt = f'''
    You are a Targeting Generator with the role "Targeting Generator".
    Generate Microsegments and Apply Specs previously stored.
    GOALS: {", ".join(goal)}
    INSTRUCTION: 
    - {"; ".join(instruction)}
    CONSTRAINTS :
    - {"; ".join(constraints)}
    
    Using the information provided, generate Microsegments for the target product: {target_product} and apply the relevant specifications.
    '''

    st.session_state["messages"].append({"role": "assistant", "content": targeting_generator_prompt})

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
