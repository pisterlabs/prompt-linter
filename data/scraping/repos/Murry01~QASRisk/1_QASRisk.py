import streamlit as st
import openai
from train_cypher import examples
from driver import read_query


# Set openAi key
openai.api_key = st.secrets["api_secret"]

with st.sidebar:
    "[View the source code](https://github.com/Murry01/QASTRisk)"
    "[Contact](https://cmvi.knu.ac.kr/)"
    st.info(
        """Example Questions:
1. List the works in the tunnel project? 
2. What are the risks associated with the tunnel project? 
3. What are the risk factors associated with the tunnel project?
4. List risks associated with the Geotechnical Investigation of the tunnel project.
5. What are the risk factors of Excavation  in the tunnel project?
6. How many risks are in the project? 

 """
    )


st.title("Question-Answering System for Construction Risks :red[(QASRisk)]")

st.markdown(
    "##### :red[A Question-Answering System for Identifying Risks in Construction Projects]"
)


# Generating Cypher Qeury from Users' Input.
def generate_response(prompt, cypher=True):
    if cypher:
        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=examples + "\n#" + prompt,
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.5,
        )
        cypher_query = completions.choices[0].text
        message = read_query(cypher_query)
        return message, cypher_query


# Form for the Input box and Submit botton
with st.form("my_form", clear_on_submit=False):
    a, b = st.columns([4, 1])
    user_query = a.text_input(
        label=":red[Ask QASGPT questions]",
        placeholder="Ask QASTRisk about construction risks...",
        label_visibility="collapsed",
    )
    # b.form_submit_button("Ask Question", use_container_width=True, type="primary")
    search_button = b.form_submit_button(
        "Ask Question", use_container_width=True, type="primary"
    )


if "generated" not in st.session_state:
    st.session_state["generated"] = []


def main():
    """
    This function gets the user input, pass it to ChatGPT function and
    displays the response
    """
    # Get user input

    # user_query = st.text_input(
    #     "Enter your queestion here",
    #     placeholder="Ask QASTRisk about construction risks...",
    # )

    with st.spinner(text="Retrieving answer...."):
        if search_button:
            # if st.button(label="Ask Question!", type="primary"):
            if user_query:
                # Pass the query to the ChatGPT function
                message, cypher_query = generate_response(user_query)
                # Display the generated Cypher query
                st.text_area("Generated Cypher Query", cypher_query, height=50)

                for i, item in enumerate(message):
                    st.write(f"{i+1}. {item}")

            else:
                st.warning("Please enter a question!")


# call the main function
main()
