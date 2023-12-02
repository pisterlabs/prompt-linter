import streamlit as st
from dotenv import load_dotenv
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
load_dotenv()

st.set_page_config(page_title="Google Search Synthesis", page_icon="🔍")

with st.sidebar:
    st.markdown('''
    ### 🔍 Google Search Synthesis: Merge the vastness of Google search results into concise, actionable insights.
- perform a google search here 
- get a precise and insightful summary of your search
- no more combing through pages of search results
- enjoy and support me with a star on [Github](https://www.github.com/Azzedde)
                    
        ''')
    

def main():
    st.header("Google Search Synthesis 🔍")
    query = st.text_input("Enter your query here")
    if query != "":
        llm = OpenAI(temperature=0)
        tools = load_tools(["serpapi", "llm-math"], llm=llm)
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        response = agent.run(query)
        st.write(response)

if __name__ == "__main__":
    main()

    

    






