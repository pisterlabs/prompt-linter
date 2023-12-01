import streamlit as st
from streamlit_chat import message
from streamlit import session_state

st.set_page_config(page_icon=":computer:", layout = "centered") # layout = "wide"
st.write("<div style='text-align: center'><h1><em style='text-align: center; color:#00FFFF;'>Interview Practice</em></h1></div>", unsafe_allow_html=True)
# ------------------------------------------------------------------------------------
import openai
import pinecone
import os

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT_NAME = st.secrets["PINECONE_ENVIRONMENT_NAME"]

openai.api_key = OPENAI_API_KEY # for open ai
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY # for lang chain

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENVIRONMENT_NAME  # next to api key in console
)
# ------------------------------------------------------------------------------------
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

#----------------------------------------------------------
from langchain.chains import RetrievalQA
from langchain import OpenAI

#defining LLM
llm = OpenAI(temperature=0)
#----------------------------------------------------------
from langchain.vectorstores import Pinecone

index_name = "customer-service-representative"
docsearch = Pinecone.from_existing_index(index_name, embeddings)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

#----------------------------------------------------------
if "asked_questions" not in st.session_state:
    st.session_state["asked_questions"] = """Below are already asked questions, dont use them again:"""
# asked_questions = """Below are already asked questions, dont use them again:"""

if "prev_template" not in st.session_state:
    st.session_state["prev_template"] = """Answer given by user for the above question is:"""
# prev_template = """Answer given by user for the above question is:"""

if "question_answer_pair" not in st.session_state:
    st.session_state["question_answer_pair"] = []
# question_answer_pair = []

if "suggestions" not in st.session_state:  
    st.session_state["suggestions"] = []
# suggestions = []

if "prev_ans" not in st.session_state:
    st.session_state["prev_ans"] = ""
# prev_ans = prev_template + "\n" + first_ans

if "question" not in st.session_state:
    st.session_state["question"] = ""

#----------------------------------------------------------
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ''

if 'userans' not in st.session_state:
    st.session_state.userans = ''

#----------------------------------------------------------

chat_col,temp = st.columns([10,1])
col1,col2 = st.columns([1,1])
text_col, temp = st.columns([10,1])
report_col,temp = st.columns([10,1])

#----------------------------------------------------------
def get_suggestions(qa_pair):
    report = "Below given is the question asked by you and the answer given by the user:\n"
    report += f"Question:{qa_pair[0]}\nAnswer:{qa_pair[1]}\n"
    report += "Now for the above qiven question and answer, give a score for the answer on a scale of 1 to 10. Also give suggestions on how to improve the user's answer."
    
    result_2 = qa({"query": report})
    suggestion_txt = result_2['result']
    
    return suggestion_txt

#----------------------------------------------------------
def submit():
    st.session_state.userans = st.session_state.input
    st.session_state.input = ''
    st.session_state["prev_ans"] = st.session_state["prev_template"] + "\n" + st.session_state.userans
    st.session_state["question_answer_pair"].append([st.session_state["question"], st.session_state["userans"]])
    st.session_state["past"].append(st.session_state["userans"])
    st.session_state["suggestions"].append(get_suggestions(st.session_state["question_answer_pair"][-1]))
    # with a_col:
    #     st.write(st.session_state.userans)

#----------------------------------------------------------
def next_question():
    query_template = f"""Give any one of the questions the Interviewer asked or cross question based on the users previous answer.
Also if user does not know the answer of a certain question move on to next question: {st.session_state["asked_questions"]} {st.session_state["prev_ans"]}"""
    
    result_1 = qa({"query": query_template})
    question = result_1['result']
    return question

#----------------------------------------------------------
def chat():
    with col1:
        if st.button("New Question"):
            st.session_state["question"] = next_question()
            st.session_state["asked_questions"] += "\n" + st.session_state["question"]
            st.session_state["past"].append(st.session_state["question"])
            
            # with q_col:
            #     st.write(st.session_state["question"])

    # for ind,i in enumerate(reversed(st.session_state["past"])):
    with chat_col:
        for ind,i in enumerate(st.session_state["past"]):
            # st.write(i)
            
            if(ind%2==0):
                message(i, key=str(i))
            else:
                message(i, is_user=True, key=str(i) + '_user')
            # with q_col:
            #     st.write(i[0])
            # with a_col:
            #     st.write(i[1])

    with col2:
        if st.button("Show Report"):
            with report_col:
            # print("Report:")
                st.write("Report:")
                for i in range(len(st.session_state["suggestions"])):
                    # print(f"Q.{i+1} {st.session_state['question_answer_pair'][i][0]}\nAns: {st.session_state['question_answer_pair'][i][1]}\nSuggestion:{st.session_state['suggestions'][i]}\n")
                    st.write(f"Q.{i+1} {st.session_state['question_answer_pair'][i][0]} \n\n Ans: {st.session_state['question_answer_pair'][i][1]} \n\n Suggestion:{st.session_state['suggestions'][i]} \n\n")
    with text_col:
        input_text = st.text_input("You: ", st.session_state["input"], key="input",
                                placeholder="Enter your response here...", 
                                label_visibility='hidden',on_change=submit)

#----------------------------------------------------------------#
def chat_customer_service_representative():
    chat()

#----------------------------------------------------------------#
# Future Roles:
# job_roles = ["Software Developer",
#              "Sales Representative",
#              "Marketing Manager", 
#              "Data Scientist", 
#              "Human Resources Manager",
#              "Project Manager",
#              "Financial Analyst",
#              "Customer Service Representative",
#              "Graphic Designer",
#              "Healthcare Administrator",
#              "Lawyer",
#              "Teacher",
#              "Web Developer"]

job_roles = ["Customer Service Representative"]

#----------------------------------------------------------------#
def main():
    st.sidebar.write("Choose a role to be interviewed for:")
    options = st.sidebar.radio("Select Role",job_roles,label_visibility="collapsed")
    
    if options == "Customer Service Representative":
        chat_customer_service_representative()

    if st.sidebar.button("Clear Chat"):
        st.session_state["asked_questions"] = """Below are already asked questions, dont use them again:"""
        st.session_state["prev_template"] = """Answer given by user for the above question is:"""
        st.session_state["question_answer_pair"] = [] 
        st.session_state["suggestions"] = []
        st.session_state["prev_ans"] = ""
        st.session_state["question"] = ""
        st.session_state["past"] = []
        st.session_state.userans = ''
        
    st.sidebar.success("Press 'New Question' to start the interview.")
#----------------------------------------------------------------#
if __name__ == "__main__":
    main()