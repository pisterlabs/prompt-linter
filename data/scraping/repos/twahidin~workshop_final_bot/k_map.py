import streamlit as st
import openai
from plantuml import PlantUML
from authenticate import return_api_key
from streamlit.components.v1 import html
import os
import re

# Create or check for the 'database' directory in the current working directory
cwd = os.getcwd()
WORKING_DIRECTORY = os.path.join(cwd, "database")

if not os.path.exists(WORKING_DIRECTORY):
	os.makedirs(WORKING_DIRECTORY)

if st.secrets["sql_ext_path"] == "None":
	WORKING_DATABASE= os.path.join(WORKING_DIRECTORY , st.secrets["default_db"])
else:
	WORKING_DATABASE= st.secrets["sql_ext_path"]

if "svg_height" not in st.session_state:
    st.session_state["svg_height"] = 1000

if "previous_mermaid" not in st.session_state:
    st.session_state["previous_mermaid"] = ""



def mermaid(code: str) -> None:
    html(
        f"""
        <pre class="mermaid">
            {code}
        </pre>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        height=st.session_state["svg_height"] + 50,
    )


def map_creation_form():
    """
    Creates a Streamlit form to collect user input for the knowledge map.
    
    Returns:
        tuple: subject, topic, levels 
    """
    
    subject = st.text_input("Enter a subject:")
    topic = st.text_input("Enter a topic to create a knowledge map:")
    levels = st.slider("Enter the number of map levels:", 1, 5, 2)
    
    if st.button('Step 1. Generate knowledge map syntax'):
        if not topic:
            st.error('Please input a topic')
        else:
            return subject, topic, levels
    return None, None, None

def map_prompter(subject, topic, levels):
    """
    Generates a prompt based on the provided subject, topic, and levels.
    
    Args:
        subject (str): Subject input by user.
        topic (str): Topic input by user.
        levels (int): Levels input by user.

    Returns:
        str: Generated prompt
    """    
    prompt = f"""Let's start by creating a diagram using the mermaid js syntax on the subject of {subject} on the topic of {topic}.
                 You must give a mindmap, class diagram or flowchart diagram in mermaid js syntax. Keep it structured from the core central topic branching out to other domains and sub-domains.
                 Let's go to {levels} levels to begin with. 
                 Expand the branch based on the complexity of each topic in terms of the time it takes to learn that topic for a beginner.You must output between these brackets with * and & as shown here for example:  *(& MERMAID SYNTAY &)*"""
    
    return prompt

def extract_mermaid_syntax(text):
    #st.text(text)
    pattern = r"```\s*mermaid\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    else:
        pattern = r"\*\(&\s*([\s\S]*?)\s*&\)\*"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return "Mermaid syntax not found in the provided text."



def map_prompter_with_mermaid_syntax(bot_response):
    """
    Generates a prompt based on a response from a chatbot for Mermaid diagram.
    
    Args:
        bot_response (str): Response from a chatbot over a topic.

    Returns:
        str: Generated prompt
    """
    
    prompt = f"""Given the insights from our chatbot: '{bot_response}', 
                 let's create a visual representation. Generate a diagram using the Mermaid JS syntax. 
                 This can be a mindmap, class diagram, or flowchart. 
                 Structure it from the central topic, branching out to other domains and sub-domains.
                 Expand the branch based on the complexity of each topic in terms of the time it takes to learn that topic for a beginner.
                 You must output the mermaid syntax between these special brackets with * and &: *(& MERMAID SYNTAX &)*"""
    
    return prompt



def generate_mindmap(prompt):
    
    try:
        openai.api_key = return_api_key()
        os.environ["OPENAI_API_KEY"] = return_api_key()
        # Generate response using OpenAI API
        response = openai.ChatCompletion.create(
                                        model=st.session_state.openai_model, 
                                        messages=[{"role": "user", "content": prompt}],
                                        temperature=st.session_state.temp, #settings option
                                        presence_penalty=st.session_state.presence_penalty, #settings option
                                        frequency_penalty=st.session_state.frequency_penalty #settings option
                                        )
        if response['choices'][0]['message']['content'] != None:
            msg = response['choices'][0]['message']['content']
            st.text(msg)
           
            extracted_code = extract_mermaid_syntax(msg)
            st.write(extracted_code)
            return extracted_code
            
            

    except openai.APIError as e:
        st.error(e)
        st.error("Please type in a new topic or change the words of your topic again")
        return False

    except Exception as e:
        st.error(e)
        st.error("Please type in a new topic or change the words of your topic again")
        return False

def output_mermaid_diagram(mermaid_code):
    """
    Outputs the mermaid diagram in a Streamlit app.
    
    Args:
        mermaid_code (str): Mermaid code to be rendered.
    """
    
    if mermaid_code:
        mermaid(mermaid_code)
    else:
        st.error("Please type in a new topic or change the words of your topic again")
        return False
     


def map_prompter_with_plantuml_form(subject, topic, levels):
    """
    Generates a prompt based on a response from a chatbot for plantuml.
    
    """
    
    prompt =  prompt = f"""Let's start by creating a simple MindMap on the subject of {subject} with topic of {topic}. 
            Can you give the mindmap in PlantUML format. Keep it structured from the core central topic branching out to other domains and sub-domains. 
            Let's go to {levels} levels to begin with. Add the start and end mindmap tags and keep it expanding on one side for now. 
            Also, please add color codes to each node based on the complexity of each topic in terms of the time it takes to learn that topic for a beginner. Use the format *[#colour] topic. 
            """
    
    return prompt

def map_prompter_with_plantuml(response):
    """
    Generates a prompt based on a response from a chatbot for plantuml.

    """
    
    prompt =  prompt = f"""Let's start by creating a simple MindMap on the chatbot response which is {response}. 
            Can you give the mindmap in PlantUML format. Keep it structured from the core central topic branching out to other domains and sub-domains. 
            Let's go to 3 levels to begin with and up to 6 at most. Add the start and end mindmap tags and keep it expanding on one side for now. 
            Also, please add color codes to each node based on the complexity of each topic in terms of the time it takes to learn that topic for a beginner. Use the format *[#colour] topic. 
            """
    
    return prompt

def generate_plantuml_mindmap(prompt):
    
    try:
        openai.api_key = return_api_key()
        os.environ["OPENAI_API_KEY"] = return_api_key()
        # Generate response using OpenAI API
        response = openai.ChatCompletion.create(
                                        model=st.session_state.openai_model, 
                                        messages=[{"role": "user", "content": prompt}],
                                        temperature=st.session_state.temp, #settings option
                                        presence_penalty=st.session_state.presence_penalty, #settings option
                                        frequency_penalty=st.session_state.frequency_penalty #settings option
                                        )
        if response['choices'][0]['message']['content'] != None:
            msg = response['choices'][0]['message']['content']
            
            p_syntax = re.search(r'@startmindmap.*?@endmindmap', msg, re.DOTALL).group()
            modified_syntax = re.sub(r'(\*+) \[', r'\1[', p_syntax)
            return modified_syntax
            

    except openai.APIError as e:
        st.error(e)
        st.error("Please type in a new topic or change the words of your topic again")
        return False

    except Exception as e:
        st.error(e)
        st.error("Please type in a new topic or change the words of your topic again")
        return False

# Define a function to render the PlantUML diagram
def render_diagram(uml):
    p = PlantUML("http://www.plantuml.com/plantuml/img/")
    image = p.processes(uml)
    return image


