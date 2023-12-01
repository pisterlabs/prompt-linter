import os
from dotenv import load_dotenv

from langchain import PromptTemplate, LLMChain  
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import re
from langchain.docstore.document import Document


from langchain.schema import SystemMessage
from fastapi import FastAPI
from agent_executor_centralized_local import PlanAndExecute
from planner.chat_planner_local import load_chat_planner
from executor.agent_executor_local import load_agent_executor
# from langchain_experimental.plan_and_execute import 
# 
#  load_agent_executor,load_chat_planner


from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain import LLMMathChain
import pdfkit
import markdown2
from md2pdf.core import md2pdf


load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


# 1. Tool for search
# Tools
llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

def save_to_pdf(markdown_content, output_filename="report.pdf"):
    css_path = "./pdf_style.css"
    # Convert the markdown content to PDF using md2pdf
    md2pdf(output_filename, md_content=markdown_content,css_file_path=css_path)

def search(objective: str, query: str):
    print("Searching...")
    # return f'search("{query}")'
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
        "num": 5
          #limit the number of search results
    })
    if query=="" or query==None or query=="string":
        return "Cannot search empty query"
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    # Acc
    # ess the "organic" field
    parsed_dict = json.loads(response.text)
    organic_results = parsed_dict.get("organic", [])

    # print("--------Organic Results--------")
    # print(organic_results)
    # print("--------END Organic Results--------")
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    prompt_template = """
    Base on the user request for this search: "{objective}",
    - Extract all relevant information from all snippets from the search results.(Cite the source of the information) put this under Key infomration section in output.
    - Base on the snippet and title of the link, predict should the content inside the url be highly valuable for the user request or not.
        -- if the answer is yes, put the url under Scrape section in output
    - output format:
    ```Key information:
        - xxx
        - (source: url1, url2)
       Scrape:
       -  [url]
       -  [None] if nothing to scrape
    ```
    Search results:
    ```{results}```

    ouput:
    """
     

    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    summarized_result=llm_chain.run({"objective": objective, "results": organic_results})
    return summarized_result


# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }
    print('URL---:'+url)
    # Convert Python object to JSON string
    data_json = json.dumps(data)
    

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)
    # print(response.content)


    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        for script in soup.find_all("script"):
            script.decompose()
        for style in soup.find_all("style"):
            style.decompose()
        for tag in soup.find_all("li", class_="menu-item"):
            tag.extract()   
        for select_tag in soup.find_all("select"):
            select_tag.decompose()         
        for span_tag in soup.find_all("span", class_="section-header"):
            span_tag.decompose()                 

        # Remove footer and image related content
        for footer_tag in soup.find_all("footer"):
            footer_tag.decompose()
        for img_tag in soup.find_all("img"):
            img_tag.decompose()
        
        text = soup.get_text()
        text = re.sub(r'\n+', '\n', text)
        if len(text) > 10:
            output = summary(objective, text[:50000])
            return output
            # return text[:3000]
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

def split_content(text):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )
    text_chunks = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in text_chunks]
    return docs

def summary(objective, content):
    print("summarization...")
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Base on the user request: "{objective}"
    Extract all relevant information from the following text 
    that will answer the user request (keep in mind this is data from scaped site so there maybe tables and other ill forated text):
    ```{text}```
    OUTPUT:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    # summary_chain = load_summarize_chain(
    #     llm=llm,
    #     chain_type='map_reduce',
    #     map_prompt=map_prompt_template,
    #     combine_prompt=map_prompt_template,
    #     verbose=True
    # )
    chain = LLMChain(llm=llm, prompt=map_prompt_template)
    output=""
    for doc in docs:
        output=output+chain.run({"text":doc, "objective": objective})
    # output = summary_chain.run(input_documents=docs, objective=objective)
    print("summarization successful...")

    return output

class SearchInput(BaseModel):
    """Inputs for search"""
    objective: str = Field(
        description="The objective of the current step")
    query: str = Field(description="search query for Google")
    
class SearchTool(BaseTool):
    name = "search"
    description = """useful for when you need to search for information on Google for a given objective.You need 2 inputs for this function. use media_type:XXX if you need to search specific type of content """
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, objective:str, query: str):
        return search(objective, query)

    def _arun(self, objective:str, query: str):
        raise NotImplementedError("error here")



class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")

#Only use this for most relevant site. 
class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = """useful to get more content inside an url and extract key information, passing both url and objective to the function; DO NOT make up any url"""
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")

def analyze_content( goal: str, content: str):
    return f'analysis'

class AnalyzeContentInput(BaseModel):
    """Inputs for analyze_content"""
    goal: str = Field(description="The goal of the analysis")
    content: str = Field(description="The content to be analyzed")


class AnalyzeContentTool(BaseTool):
    name = "analyze_content"
    description = "useful when you need to analyze some information, pass both the goal of the analyssi, and the content in string to the function."
    args_schema: Type[BaseModel] = AnalyzeContentInput

    def _run(self, goal: str, content: str):
        # print("HERE!!!!"+"{goal},{content}")
        return analyze_content(goal, content)

    def _arun(self, content: str):
        raise NotImplementedError("error here")



    #ScrapeWebsiteTool(),

tools = [
    SearchTool(),
    ScrapeWebsiteTool(),
]



global expander
# 4. Use streamlit to create a web app
def main():

    st.set_page_config(
    page_title="Google Search and Researcher",
    # layout="wide"
    )        
    st.header("Let AI help you with Google research")

    st.markdown('<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto:400,100,300,500,700,900|Roboto+Condensed:400,700&display=swap" />', unsafe_allow_html=True)
    st.markdown(f'''
        <style>
        header {{
        font-family: 'Roboto', sans-serif;   
        text-align: center;
        font-size: 42px;
        font-weight: 600;
        color: #FF4C29;
        }}
        input {{
        font-size: 20px;
        padding: 10px;
        outline: none;
        border: 2px solid #FF4C29;
        border-radius: 10px;
        }}  
        </style>
        ''', unsafe_allow_html=True) #"top 100 baby names in US" ,"Active venture investors investing in AI in 2023" 
    

    query=st.text_input("Searching for")
    
    # Initialize session state variables if not present
    if "search_triggered" not in st.session_state:
        st.session_state.search_triggered = False
    if "summarized_result" not in st.session_state:
        st.session_state.summarized_result = ""
    if "report_generated" not in st.session_state:
        st.session_state.report_generated = False
    if "stop_research" not in st.session_state:
        st.session_state.stop_research = False        

    col1, col2 = st.columns(2)




    if col1.button('Generate Report') and (query and st.session_state.search_triggered == False):        
        st.session_state.search_triggered = True
        stop_button = col2.button("Stop")
        if stop_button:
            st.session_state.stop_research = True
        model4 = ChatOpenAI(temperature=0,model_name="gpt-4")
        model = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo-16k-0613")
        planner = load_chat_planner(model4,tools)
        st.write("Doing research for ", query)
        
        # Use the planner to get the planned steps
        inputs = {
            "input": query,
        }
        planned_steps = planner.plan(inputs)
        status=st.status('Planed Research Area',expanded=True,state='running')
        for p_step in planned_steps.steps:
            print("-",p_step.value,'\n')
            status.write('\u2192    ' + p_step.value, expanded=False)
        status.update(expanded=False,state='complete')            
        if st.session_state.stop_research:
            st.warning("Research process halted by the user.")
            st.session_state.search_triggered = False
            st.session_state.stop_research = False
            st.stop()        
        executor =load_agent_executor(model, tools, verbose=True)
        agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)     
        response=agent.run(inputs)
        print(response)
        print("\n\nCompling final results")
        prompt_template = """
        Base on the objective for this research: "{objective}",
        - write the following sections of a reserach report: 
        -- "Executive Summary (draw conclusion from observation and summarize in less than 3 bullet points in answering the objective directly and precisely: be short, cociese, use reference url)
        -- "Analysis and Interpretation" (analyze the information based on patterns you've noticed)"
        - Reference source url throughout the paragraph when appropriate
        - Write the reports using markdown style categorizing information so that it is easy for the user to read.
        - Use table with 2+ colummns to organize information so it's easier to read.
        - Don't make up fact, use fact from the information you've found.
         ```{results}```
        """
        llm_chain = LLMChain(
        llm=model,
        prompt=PromptTemplate.from_template(prompt_template)
        )
        summarized_result_part1=llm_chain.run({"objective": query, "results": response.steps})

        prompt_template = """
        Base on the objective for this research: "{objective}",
        - write the following sections as part of an overall report(no need to write title for the report): 
        -- Detail findings (break down in sub-sections if needed, brief the observations in details for the sub-section topic,reference source articles), 
        -- References (list of referenced resources and its name) 
        -- Method (list the research steps you've taken for this report in short bullet point)",        
        - Reference source url throughout the paragraph when appropriate
        - Write sections using markdown style categorizing information so that it is easy for the user to read.
        - Use table with 2+ colummns to organize information so it's easier to read.
        - Don't make up fact, use fact from the resources you found.
         ```{results}```
        """
        llm_chain = LLMChain(
        llm=model,
        prompt=PromptTemplate.from_template(prompt_template)
        )
        summarized_result_part2=llm_chain.run({"objective": query, "results": response.steps})
        summarized_result=summarized_result_part1+'\n'+summarized_result_part2
        save_to_pdf(summarized_result)
        st.session_state.report_generated = True

        print("Compling final results successful")
        # print(summarized_result)
        # st.info(summarized_result)    

        st.session_state.summarized_result = summarized_result

    if st.session_state.report_generated:
        with open("report.pdf", "rb") as f:
            bytes = f.read()
            st.download_button(
                label="Download PDF",
                data=bytes,
                file_name=query.replace(' ','_')+".pdf",
                mime="application/pdf"
            )
        # Display the planned steps on Streamlit
        st.session_state.report_generated = False
    if "summarized_result" in st.session_state:
        st.info(st.session_state.summarized_result)
        # st.info(planned_steps)
    st.session_state.search_triggered = False
    st.session_state.report_generated = False

if __name__ == '__main__':
    main()
##top AI investors who are actively investing in AI in 2023





### TODO: 
##Memory needs to be flushed after each step and we only keep teh final answer from each step as an input to the next step 

##Memory integrate a long term memroy which can be just a long string of text
##seperate report formatting into a different tool 