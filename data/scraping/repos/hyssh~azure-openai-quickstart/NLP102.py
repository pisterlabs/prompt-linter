import os
import openai
import dotenv
import pandas as pd
import streamlit as st
from sqldbconn import get_connx
from StepAutomation import StepAutomation

# .env file must have OPENAI_API_KEY and OPENAI_API_BASE
dotenv.load_dotenv()
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

def sample102():
    stepautomation = StepAutomation()

    # create a function checking stepautomation get current_stepautomation_structure_step in a while loop
    def stepautomation_process(stepautomation : StepAutomation, system_msgs: str, question: str):
        """
        This function will update the stepautomation process until it reaches the final answer.
        """
        query = ""

        # Step 1. send the question and get the thought process 1
        assistant_question_res = stepautomation.run(user_msg={"role":"user", "content":"Question:\n" + question + "\nThought Process 1 Start:\n"} )
        with st.expander("See Thought Process 1"):
            st.info(assistant_question_res)

        # Step 2. send the thought process 1 and get the query
        try:
            query = stepautomation.run(user_msg={"role":"user","content":"Writre T-SQL Query Start:\n"},
                                        temperature=0.7,
                                        max_tokens=400).split("```")[1]
            with st.expander("See T-SQL Query"):
                st.info(query)
        except IndexError:
            st.error("Failed to extract T-SQL query from OpenAI response.")
            return "I am sorry, I don't understand your command. Can you please rephrase your command?"

        # Step 3. send the query and get the result    
        df = pd.read_sql_query(query, get_connx())
        with st.expander("See Result"):
            st.dataframe(df.head(10))

        # Step 4. send the result and get the thought process 2    
        assistant_query_df_msg = {"role":"assistant","content":"Writre T-SQL Query Start: \n```\n" + query +"```\nWritre T-SQL Query End: " + "Result Start:"+df.head(10).to_json()+"Result End:\n"}
        thought_process_2_res = stepautomation.run(user_msg={"role":"user","content":"Thought Process 2 Start:\n"},
                                            assistant_msg=assistant_query_df_msg, 
                                            temperature=0.7,
                                            max_tokens=500)
        with st.expander("See Thought Process 2"):
            st.info(thought_process_2_res)

        # Step 5. send the thought process 2 and get the thought process 3
        thought_process_3_res = stepautomation.run(user_msg={"role":"user","content":"Thought Process 3 Start:\n"}, 
                                            temperature=0.7,
                                            max_tokens=500)
        with st.expander("See Thought Process 3"):
            st.info(thought_process_3_res)

        # Step 6. send the thought process 3 and get the final answer
        final_answer_res = stepautomation.run(user_msg={"role":"user","content":"Final Answer Start:\n"},
                                            temperature=0.7,
                                            max_tokens=1000)
        return final_answer_res

    st.markdown("# Generate Additional Insights using OpenAI")
    st.markdown("OpenAI review data and provides additional insights to the user")
    with st.expander("Demo scenario"):
        st.image("https://github.com/hyssh/azure-openai-quickstart/blob/main/images/Architecture-demo.png?raw=true")
        st.markdown("1. User will type a question in the input box")
        st.markdown("2. __Web App__ sends the question to __Azure OpenAI__")
        st.markdown("3. __Azure OpenAI__ will convert the question to SQL query")
        st.markdown("4. __Web App__ sends the SQL query to __Azure SQL DB__")
        st.markdown("5. __Azure SQL DB__ will execute the SQL query and return the result to __Web App__")
        st.markdown("6. __Web App__ will show the result to user")
    
    st.markdown("---")
    with st.container():
        # create a input using streamlit
        inputmsg = st.text_input("Type your question about World Wide Importers database here")

    with st.sidebar:
        with st.container():
            st.info("This sample use chain of thought to get the final answer")
        sample_tab, prompt_tab, system_tab, stepautomation_tab = st.tabs(["samples", "prompts", "system", "Chain of Thought"])

        # samples
        with sample_tab:
            st.header("Samples")
            st.code("high-value customers for targeted marketing campaigns or loyalty programs", language="html")
            st.code("average number of orders and purchase amount per customer ", language="html")
            st.code("Get top 5 most popular products with the amount of the purchases", language="html")
            st.code("Get customer email and address who purchased our products the most with the amount of the purchases", language="html")
            st.code("Create a list combining product category and product naming it 'ProductShortDesc' name and add sales revenue", language="html")
            st.code("Drop Customer table", language="html")

        # prompts
        with prompt_tab:
            st.header("Prompts")
            st.code(stepautomation.database_summary, language="html")

        with system_tab:
            st.header("System")
            st.code(stepautomation.system_char, language="html")
        
        with stepautomation_tab:
            st.header("Chain of Thought")
            st.code(stepautomation.instruction_prompt, language="html")

    # if input is not empty, then run the code below
    if inputmsg:
        with st.spinner("Loading..."):
            st.write(stepautomation_process(stepautomation, stepautomation.get_history(), inputmsg))
            st.markdown("---")
            with st.expander("See Prompt"):
                st.markdown("## Prompt")
                st.write(stepautomation.get_history())

if __name__ == "__main__":
    sample102()