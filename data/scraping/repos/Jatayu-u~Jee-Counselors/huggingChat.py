import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from dotenv import load_dotenv
# Importing openai
import openai

# IMPORTING THE LIBRARIES FOR LANGCHAIN AGENT
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper

# LIBRARIES FOR USING LLM CHAIN WITH AGENTS
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
openai_api_key = "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"
google_api_key = "AIzaSyAFjcF_fTDyzrvpeA8dYgMJKM0SCUt5Hss"

google_cse_id = "240babe5bb908429d" 
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re



# Loading api calls





# load the Environment Variables. 
# load_dotenv()

search = GoogleSearchAPIWrapper()

st.set_page_config(page_title="OpenAssistant Powered Chat App")

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 JEE Counselor')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot to help jee aspirants with their doubts about preparation and resources.
    ''')
    add_vertical_space(3)
    st.write('Made with ❤️ by Shaurya Vats (Indian Institute of Technology, Kharagpur)')

st.header("Your Personal Counselor 💬")

def main():
            # SETTING THE MEMORY OF THE CHAT HISTORY
    memory = ConversationBufferMemory(memory_key='chat_history')        
    if "memory" not in st.session_state:
        st.session_state.memory = memory
    
    # Generate empty lists for generated and user.
    ## Assistant Response
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["I'm here to guide you, How may I help you today?"]

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


        # TOOLS THAT WILL USED BY THE AGENTS
        tools = [
    Tool(
        name = "Web Search",
        func=search.run,
        description="Used when any fact based question is asked and resources are to be shared from the internet."
    )]
        
        # PROMPT TEMPLATE FOR THE AGENT TO ACT AS A JEE C0UNSELOR.
        
        template = """
        Act as a JEE counselor, entrusted with the responsibility of guiding students to improve their performance in JEE Mains and Advanced. Your role is to provide comprehensive assistance to each student query in a manner that promotes their growth and success. Here are the key guidelines to ensure an effective counseling experience:

Respond to student queries with the utmost politeness and encouragement. Maintain a positive and supportive tone throughout the conversation. Your aim is to inspire and motivate students to reach their full potential
List all the available online resources and study materials that can aid students in their JEE preparation. Prioritize reliable and reputable sources that align with the JEE syllabus. Ensure that students have access to a wide range of materials for comprehensive learning.
 Before providing guidance, thoroughly explore all the available resources. Familiarize yourself with study materials, online courses, practice tests, and other relevant resources that can enhance students' preparation. Your responses should reflect the knowledge and insights gained through this exploration.
Incorporate motivating quotes and examples in your responses to inspire and uplift students. Share success stories of individuals who have achieved remarkable results in JEE. Additionally, draw from your personal experiences and provide advice as if you have gone through similar situations. This will create a relatable and empathetic connection with the students.
If a question unrelated to JEE, life, or counseling is asked, kindly redirect the conversation by asking questions on these relevant topics instead. Engage the students in meaningful discussions related to JEE preparation, share insights about life as a JEE aspirant, or offer counseling on academic and personal growth.
Provide as much information, facts, and stories about the questions and colleges as possible. Ensure that your responses include detailed insights and knowledge about the specific topics at hand. Students should benefit from your comprehensive understanding of the subject matter.
Remember to look into all the available resources before answering any query. By following these guidelines, you can effectively fulfill your role as a JEE counselor and empower students to improve their performance in JEE Mains and Advanced. Your responses should not only be polite and encouraging but also reflect your personal experiences, include motivating quotes and examples, and offer comprehensive information to support students' growth and success.


        You have the access of the following tools:
                 {tools}
 

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, could be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        

        Previous conversation history:
        {chat_history}

        Question: {question}
        
        {agent_scratchpad}
                        
       
        """
        
        
        
        
        
        
        # SETTING UP ALL THE OUTPUT AND INPUT FUNCTIONS TO USE LLM CHAIN WITH AGENT TAKEN FROM DOCUMENTATIONS
        # Set up a prompt template
        class CustomPromptTemplate(StringPromptTemplate):
            # The template to use
            template: str
            # The list of tools available
            tools: List[Tool]

            def format(self, **kwargs) -> str:
                # Get the intermediate steps (AgentAction, Observation tuples)
                # Format them in a particular way
                intermediate_steps = kwargs.pop("intermediate_steps")
                thoughts = ""
                for action, observation in intermediate_steps:
                    thoughts += action.log
                    thoughts += f"\nObservation: {observation}\nThought: "
                # Set the agent_scratchpad variable to that value
                kwargs["agent_scratchpad"] = thoughts
                # Create a tools variable from the list of tools provided
                kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
                # Create a list of tool names for the tools provided
                kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
                return self.template.format(**kwargs)
# SETTING UP THE PROMPT FOR THE AGENT USING ALL THE INPUT VARIABLES AND TOOLS
        prompt = CustomPromptTemplate(
            template=template,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["question", "intermediate_steps","chat_history"]
        )
        # SETTING UP THE OUTPUT PARSER FOR THE OUTPUT PRODUCED BY THE LLM
        class CustomOutputParser(AgentOutputParser):

            def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
                # Check if agent should finish
                if "Final Answer:" in llm_output:
                    return AgentFinish(
                        # Return values is generally always a dictionary with a single `output` key
                        # It is not recommended to try anything else at the moment :)
                        return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                        log=llm_output,
                    )
                # Parse out the action and action input
                regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
                match = re.search(regex, llm_output, re.DOTALL)
                if not match:
                    raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
                action = match.group(1).strip()
                action_input = match.group(2)
                # Return the action and action input
                return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)       
        

        # SETTING UP THE LARGE LANGUAGE MODEL FOR THE APP
        # llm=HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", model_kwargs={"max_new_tokens":1200})
        llm = OpenAI(model='text-davinci-003')
        
        # SETTING UP THE LLM CHAIN FOR THE AGENT
        llm_chain=LLMChain(
            llm=llm,
            prompt=prompt
        )
        
        # USING THE CUSTOM OUTPUT FUNCTION 
        output_parser = CustomOutputParser()
        
        tool_names = [tool.name for tool in tools]
        agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
                
        
)
        # EXECUTING THE AGENT
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True,memory = memory)
        
        return agent_executor
    
    # if 'memory' not in st.session_state:
    #     st.session_state['memory'] = ''


    # generate response
    def generate_response(question, llm_chain):
        response = llm_chain.run(question)
        return response

    ## load LLM
    llm_chain = chain_setup()

    # main loop
    with response_container:
        if user_input:
            response = generate_response(user_input, llm_chain)
            st.session_state.user.append(user_input)
            st.session_state.generated.append(response)
            # st.session_state["memory"] += st.session_state.memory.buffer
            
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['user'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))

if __name__ == '__main__':
    main()