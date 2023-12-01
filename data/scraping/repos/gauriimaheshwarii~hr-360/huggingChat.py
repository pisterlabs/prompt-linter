import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from dotenv import load_dotenv
from docx import Document

big_response=""
gauri=""

# def func(string: str):
#     global gauri+=string

# load the Environment Variables. 
load_dotenv()
st.set_page_config(page_title="OpenAssistant Powered HR")


st.header("HR 360")

def main() -> str:
    res_test=""
    
    flag=True

    # Generate empty lists for generated and user.
    ## Assistant Response
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["I'm Assistant, How may I help you?"]

    ## user question
    if 'user' not in st.session_state:
        st.session_state['user'] = ['Hi!']

    # Layout of input/response containers
    response_container = st.container()
    colored_header(label='', description='', color_name='blue-30')
    input_container = st.container()

    # get user input
    def get_text():
        input_text = st.text_input("You: ", "", key="input")
        return input_text

    ## Applying the user input box
    with input_container:
        user_input = get_text()

    def chain_setup():


        template = """<|prompter|>{question}<|endoftext|>
        <|assistant|>"""
        
        prompt = PromptTemplate(template=template, input_variables=["question"])

        llm=HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", model_kwargs={"max_new_tokens":1200})

        llm_chain=LLMChain(
            llm=llm,
            prompt=prompt
        )
        return llm_chain


    # generate response
    def generate_response(question, llm_chain):
        response = llm_chain.run(question)
        return response

    ## load LLM
    llm_chain = chain_setup()
    
    gauri=""

    # main loop
    with response_container:
        if (flag):
            if user_input:
                response = generate_response(user_input, llm_chain)
                
                # res_test+=str(response)
                gauri+=response
                
                
                # print(response)
                print(type(response))
                st.session_state.user.append(user_input)
                st.session_state.generated.append(response)
            
            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state['user'][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i))
        flag=False
    # global gauri=gauri+res_test
    return gauri


if __name__ == '__main__':
    
    tech_roles = [
    "Software Engineer",
    "Front-end Developer",
    "Back-end Developer",
    "Full-stack Developer",
    "DevOps Engineer",
    "Data Scientist",
    "Machine Learning Engineer",
    "AI Researcher",
    "Data Engineer",
    "Database Administrator",
    "System Administrator",
    "Cloud Architect",
    "Network Engineer",
    "Security Analyst",
    "Quality Assurance (QA) Engineer",
    "Web Developer",
    "Mobile App Developer",
    "UI/UX Designer",
    "Product Manager",
    "Business Analyst",
    "Project Manager",
    "Scrum Master",
    "Technical Support Engineer",
    "Network Administrator",
    "Game Developer",
    "Embedded Systems Engineer",
    "Cybersecurity Specialist",
    "Blockchain Developer",
    "Big Data Engineer",
    "Quantum Computing Researcher",
    "AR/VR Developer",
    "Bioinformatics Scientist",
    "Hardware Engineer",
    "Automation Engineer",
    "Robotics Engineer",
    "Technical Writer",
    "Data Analyst",
    "IT Consultant",
    "ERP Consultant",
    "Bioinformatics Analyst",
    "Geospatial Analyst",
    "CAD Designer",
    "GIS Specialist",
    "Business Intelligence Analyst",
    "Firmware Engineer",
    "Wireless Network Engineer",
    "Control Systems Engineer",
    "Natural Language Processing (NLP) Engineer",
    "VR/AR Artist",
    "Human-Computer Interaction (HCI) Researcher",
    "Simulation Engineer",
    "Bioinformatics Engineer",
    "Autonomous Vehicle Engineer"]
        

    x_test=main()
    # print(x_test)
    
    matching_roles=[]
    
    for role in tech_roles:
        if role in x_test:
            matching_roles.append(role)
            
    print(matching_roles)
    print("1st")
    
    def chain_setup():


        template = """<|prompter|>{question}<|endoftext|>
        <|assistant|>"""
        
        prompt = PromptTemplate(template=template, input_variables=["question"])

        llm=HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", model_kwargs={"max_new_tokens":1200})

        llm_chain=LLMChain(
            llm=llm,
            prompt=prompt
        )
        return llm_chain
    
    llm_chain = chain_setup()
    
    
    def generate_response(question, llm_chain):
        response = llm_chain.run(question)
        return response
    
    
    
    
    
    
    for i in matching_roles:
        prev="kindly generate a job description with years of expereinece required, and skills required for a professional "
        end=" in brief"
        
        net_input=prev+i
        
        resp=generate_response(net_input,llm_chain)
        print("HI")
        
        print(resp)
        
        doc = Document()


        text = resp
        
        doc.add_paragraph(text)
        
        check="job_description "+i
        doc.save(check)

        print("Document saved successfully.")
        break
        

    
    
    

