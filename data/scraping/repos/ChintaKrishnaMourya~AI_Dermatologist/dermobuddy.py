from langchain import PromptTemplate
from langchain.chains import LLMChain
import easyocr
from langchain.document_loaders import YoutubeLoader
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import DuckDuckGoSearchResults
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from langchain.agents import initialize_agent
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from textbase.models import OpenAI
import openai
from langchain.llms import OpenAI
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
import string

# nltk.download('punkt')
# nltk.download("stopwords")
# nltk.download('wordnet')

api_key= ""

def extract_text(image_path, language='en'):
    try:
        # Initialize the EasyOCR reader with the specified language
        reader = easyocr.Reader([language])

        # Read text from the image
        result = reader.readtext(image_path)

        # Extract and return the text from the result
        extracted_text = ' '.join([item[1] for item in result])

        return extracted_text
    except Exception as e:
        return str(e)

wrapper = DuckDuckGoSearchAPIWrapper(region="in-en", max_results=10)
ddgsearch = DuckDuckGoSearchResults(api_wrapper=wrapper)

def duck_search_sale(query):
    duck_search_results = ddgsearch(query)
    duck_search_results = duck_search_results.lower()
    return duck_search_results



template_ocr = """
You are an expert dermatologist. You have to provide relevant answer to 'ocr_tool_output_and_query' that you have recieved under Paragraph section below.\
you answer without any hallucinations and false information. You should ask follow up questions to users before proceeding to give the final direct answer.
Remember, You are aware that 'ocr_tool_output_and_query' is being extracted from image by using OCR tool and as such there may be some errors in the extraction.\
OCR Tool will attempt to account for some words being swapped with similar-sounding words or may also be irregular or incomplete.\
You have to understand this 'ocr_tool_output_and_query' in terms of dermatology field and try your best to answer the query by analyzing the query and understanding the ingredients used and give indepth analyis.

Paragraph
ocr_tool_output_and_query : {input}

# """
prompt_ocr = PromptTemplate(
    input_variables=["input"],
    template=template_ocr
)


chain_ocr = LLMChain(
    llm=ChatOpenAI(temperature=0, model = 'gpt-3.5-turbo', max_tokens = 3000, openai_api_key=api_key),
    prompt=prompt_ocr,
    verbose=False,
)

#TOOL
desc_ocr="""

Use this tool exclusively when user gives image url address or link. This tool extracts text from image and answers to user.

"""

class OCRTool(BaseTool):
  name='Text extracted from image'
  description=desc_ocr

  def _run(self,image_url:str)-> str:
    data=chain_ocr.run(input = extract_text(image_url))
    return data

  def _arun(self,symbol):
    raise NotImplementedError("This tool doesnt support async")

ocrtool=OCRTool()



template = """
<s>[INST] <<SYS>>

You are an expert dermatologist. You have to provide relevant answer to the user based on the 'Context' and 'Query' you recieved. Both 'Query' and 'Context' is under the Paragraph section below. You exactly only reply to the 'Query' provided below under Paragraph section without any hallucinations and false information.
Remember, You are aware that 'Context' is being extracted from the internet or the web search and as such there may be some errors.\
You have to understand this 'Context' interms of dermatology field. Please include the extensive knowledge that you posses from your training in dermatology while giving recommendation or while answering users 'Query' inaddition to the 'Context'
You have to understand this 'Query' in terms of dermatology field and try your best to answer the query by analyzing the query and understanding the ingredients that user mentioned in query and give indepth analyis.
You should ask the follow up questions if necessary to the user.
<</SYS>>

Paragraph
Query: {query}
Context: {context}
 [/INST]

"""

prompt = PromptTemplate(template=template, input_variables=["query","context"])
search_llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo', openai_api_key=api_key
)
llm_chain_search = LLMChain(prompt=prompt, llm=search_llm)

desc_search="""

Use this tool to search the internet when only the query is given. This tool accepts single parameter called user query

"""

class SearchTool(BaseTool):
  name='General_Search'
  description=desc_search

  def _run(self,query:str)-> str:
    data=llm_chain_search.run(query=query,context=duck_search_sale(query))
    return data

  def _arun(self,symbol):
    raise NotImplementedError("This tool doesnt support async")

searchtool=SearchTool()

"""## Youtube Transcript Summary Tool"""

template_youtube = """
You are an expert summarizer who has a greate expertise in dermatology. you get 'context' which is transcribed from youtube. Understand the context and summarize it in 'bullet points'.
But remember, you should only summarize the videos that are related to dermatology and nothing else.
context = {transcripts}
Summary =
"""
prompt_youtube = PromptTemplate(
    input_variables=["transcripts"],
    template=template_youtube
)


chain_youtube = LLMChain(
    llm=ChatOpenAI(temperature=0, model = 'gpt-3.5-turbo-16k', openai_api_key=api_key),
    prompt=prompt_youtube,
    verbose=False,
)

summary_desc = "Use this tool to summarize the youtube video content"
class transcript_tool(BaseTool):
  name = "youtube_video_summary"
  description =summary_desc

  def _run(self, url:str):
    try :
      loader = YoutubeLoader.from_youtube_url( url, add_video_info=True)
      transcript = loader.load()[0].page_content
      summary =chain_youtube.run(transcript)
      return summary

    except ValueError as e:
        summary = f"Invalid YouTube URL: {str(e)}"
    except Exception as e:
        summary = "Sorry, Currently we are supporting only videos that have transcripts. Please find videos that have transcripts. We are making improvements to transcribe text from audio using whisperAI or etc"
    return summary
trans = transcript_tool()



sys_msg ="""
You are an expert assistant chatbot in dermatology trained by 'DermVisionAI' for assisting users in their skin health related search queries.\
You are capable of giving expert level dermatology assistance and extracting the details from user given image of product and providing information about it and also summarziring the skin care related youtube videos. 
You can use these tools 'Text extracted from image', 'General_Search', 'youtube_video_summary'  wisely for the queries. 'General_Search' is used when user asks anything that related to skin health or dermatology.\
It has access to internet to give latest results etc.\
'youtube_video_summary' tool is used when user gives youtube link to get the content or summary of that video. But this video must belong to dermatology or else give output as "Kindly, Provide skin health related video content. I can't summarize outside of this".
'Text extracted from image' tool is used to extract text from user given image link or address.
You are constantly learning and training. You are capable of answering all dermatology related queries effectively. You never hallucinate answers, you always give authentic answers to best of your ability without any false information.
If user says Hi, respond with Hello! How can I assist you Today?
You always give indepth answers to users with detailed explanations step by step.
Do not answer any private, general questions other than dermatology related user queries
You should ask users necessary follow up questions before proceeding to use tools.

"""

tools = [

    Tool(name = "Text extracted from image",
         func = ocrtool._run,
         description = desc_ocr,
         return_direct = False

    ),

    Tool(name = "General_Search",
         func = searchtool._run,
         description = desc_search,
         return_direct = True

    ),

    Tool(name = "youtube_video_summary",
         func = trans._run,
         description = summary_desc,
         return_direct = True)
    ]
agent_llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-4',
        openai_api_key=api_key
)

# agent = initialize_agent(tools, llm = agent_llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

conversational_memory = ConversationBufferWindowMemory(
        memory_key = "chat_history",
        k = 6,
        return_messages=True,
)


derm_agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=agent_llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory,
    handle_parsing_errors=True,

)


new_prompt = derm_agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools

)

derm_agent.agent.llm_chain.prompt = new_prompt
