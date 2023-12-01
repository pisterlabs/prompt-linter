import anvil.facebook.auth
import anvil.google.auth, anvil.google.drive, anvil.google.mail
from anvil.google.drive import app_files
import anvil.users
import anvil.files
from anvil.files import data_files
import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
import anvil.secrets
import anvil.server
import anvil.http
import openai
import time
import requests
from langchain.agents import load_tools, initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain import SerpAPIWrapper, LLMChain, PromptTemplate
from langchain.tools import StructuredTool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities import SerpAPIWrapper
import json
import re
from collections import OrderedDict

import requests
from bs4 import BeautifulSoup

############################################################################################################################
openai_api_key = anvil.secrets.get_secret('OPENAI_API_KEY')
serpapi_api_key = anvil.secrets.get_secret('SERPAPI_API_KEY')
google_cse_id = anvil.secrets.get_secret('GOOGLE_CSE_ID')
google_api_key = anvil.secrets.get_secret('GOOGLE_API_KEY')
############################################################################################################################


############################################################################################################################
### TOOLS

# SERPAPI
search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
tools = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run,
)

# # GOOGLE SEARCH - DOESN'T WORK
# search = GoogleSearchAPIWrapper(google_api_key=google_api_key, google_cse_id=google_cse_id)
# tools = Tool(
#     name="Google Search",
#     description="Search Google for recent results.",
#     func=search.run,
# )



############################################################################################################################

####### -------- PRELIMINARY / FIRST DRAFTS--------###########

# COMPANY 1st DRAFT

# USE WEBSCRAPER
 
@anvil.server.callable
def launch_draft_company_summary_scraper(company_name, company_url):
    # Launch the background task
    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='company_profile_latest')
    company_dump_row = user_table.get(variable='company_page_dump')

    # START THE WEB SCRAPING
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
    page_content = requests.get(company_url, headers=headers).content

    soup = BeautifulSoup(page_content, "html.parser")
    # Extract all the text from the page
    bulky_text_content = soup.get_text()
   # Remove leading and trailing whitespaces, replace newlines and extra spaces
    company_context_scraped_bulky = bulky_text_content.strip().replace('\n', ' ').replace('\r', '').replace('  ', ' ')

    # Further remove extra white spaces
    company_context_scraped = re.sub(r'\s+', ' ', company_context_scraped_bulky.strip())
  
    print("Scraped Information:",company_context_scraped)

    print("Launch task started for researching company:",company_url)
    task = anvil.server.launch_background_task('draft_company_summary_scraper', company_name, company_url,row,company_context_scraped)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def draft_company_summary_scraper(company_name, company_url,row,company_context_scraped):
    #Perform the Webscraping
    print("Background task started for generating the company summary:", company_url)
   
    llm_agents = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo-16k', openai_api_key=openai_api_key)
    template_company_summary = """As a highly-skilled business analyst, your task is to conduct an exhaustive analysis to build an informational company profile of {company_name}. \
                    Leverage the below provided company research context scraped from the company's website {company_url}, to create a complete company profile.  \
                    
                    Lastly, be very specific! This is not an educational excercise. This work will be incorporated into our commercial operation shortly, so provide a meaningful synopsis and findings. Do not provide general terms or vague business ideas: be as particular about the issue as possible. Be confident. Provide numbers, statistics, prices, when possible!
                    \n \
                    Overview: Provide a comprehensive introduction to the company. What are the unique features or value propositions of the company's offerings? What does the company aim to achieve? \n \
                    \n \
                    Unique Value Proposition: What is the company unique value proposition? What are they uniquely positioned to do? How does their main offer differ from their competitors? \n \
                    \n \
                    Founding Story: What inspired the founders to start the company? Are there any unique or interesting anecdotes about the early days of the company? How has the company evolved since its founding? \n \
                    \n \
                    Competitors: Who are the likely competitors of this company? What are their strengths and weaknesses? How does your company compare to its competitors in terms of offerings, market share, or other relevant factors?  \n \
                    \n \                
                    Mission & Vision: What is the company's mission statement or core purpose? What are the long-term goals and aspirations of the company? \n \
                    Values: What does the company value? What do they emphasize in their mission? What do they care about or prioritize? \n \
                    \n \

                    NOTES ON FORMAT:
                    This should be at least 800 words. Be confident. If there is incomplete information, please state "MORE INFORMATION NEEDED"! Speak as if you are the authority of the subject. 
                    Ensure you keep the headers with the '--': 
                    -- Overview
                    (your overview)
                  
                    --Unique Value Proposition
                    (your response)
                    
                    --Competitors
                    (your response)
                    
                    -- Founding Story
                    (your response)
                    
                    --Mission & Vision
                    (your response)

                  --Values
                    (your response)

                  ** END OF FORMAT
                  
                  FINALLY, HERE IS THE COMPANY CONTEXT SCRAPED FROM THEIR WEBSITE: {company_context_scraped}
                    """
  
    prompt_company_summary = PromptTemplate(
        input_variables=["company_name", "company_url","company_context_scraped"],
        template=template_company_summary
    )
  
    chain_company_summary = LLMChain(llm=llm_agents, prompt=prompt_company_summary)
    draft_company_summary = chain_company_summary.run(company_name=company_name,company_url=company_url,company_context_scraped=company_context_scraped)  # Pass in the combined context

  # Save this generated version as the latest version
    row['variable_value'] = draft_company_summary
    row.update()
    print("Company Research Complete")

### THESE ARE THE COMPANY SEARCH AGENTS  
# COMPANY 1st DRAFT
# @anvil.server.callable
# def launch_draft_company_summary(user_table,company_name, company_url):
#     # Launch the background task
#     current_user = anvil.users.get_user()
#     user_table_name = current_user['user_id']
#     # Get the table for the current user
#     user_table = getattr(app_tables, user_table_name)
#     row = user_table.get(variable='company_profile_latest')
  
#     print("Launch task started for researching company:",row, company_name,company_url)
#     task = anvil.server.launch_background_task('draft_company_summary', row,company_name, company_url)
#     # Return the task ID
#     return task.get_id()
  
# @anvil.server.background_task
# def draft_company_summary(row,company_name, company_url):
#     print("Background task started for researching company:", row,company_name,company_url)
#     llm_agents = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)
#     agent_company_context = initialize_agent([tools], llm_agents, agent="zero-shot-react-description", handle_parsing_errors=True)
#     company_research = agent_company_context({"input": f"""As a highly-skilled business research agent, your task is to conduct an exhaustive analysis to build an informational company profile of {company_name}. \
#                     Leverage all necessary resources, primarily the company's website {company_url}, but also news articles, and any other relevant sources.  \
#                     to gather the following details about {company_name}.  Lastly, be very specific! This is not an educational excercise. This work will be incorporated into our commercial operation shortly, so provide meaningful research and findings. Do not provide general terms or vague business ideas: be as particular about the issue as possible. Be confident. Provide numbers, statistics, prices, when possible!
#                     \n \
#                     Overview: Provide a comprehensive introduction to the company. What are the unique features or value propositions of the company's offerings? What does the company aim to achieve? \n \
#                     \n \
#                     Unique Value Proposition: What is the company unique value proposition? What are they uniquely positioned to do? How does their main offer differ from their competitors? \n \
#                     \n \
#                     Founding Story: What inspired the founders to start the company? Are there any unique or interesting anecdotes about the early days of the company? How has the company evolved since its founding? \n \
#                     \n \
#                     Competitors: Who are the likely competitors of this company? What are their strengths and weaknesses? How does your company compare to its competitors in terms of offerings, market share, or other relevant factors?  \n \
#                     \n \                
#                     Mission & Vision: What is the company's mission statement or core purpose? What are the long-term goals and aspirations of the company? \n \
#                     Values: What does the company value? What do they emphasize in their mission? What do they care about or prioritize? \n \
#                     \n \
                  
#                     NOTES ON FORMAT:
#                     This should be at least 800 words. Be confident, do not say there is incomplete information, or there is not information. If you can't answer elements from the above, ignore it! Speak as if you are the authority of the subject. If you don't know the answer, don't talk about it. Do not say "I was unable to find information on XYZ". 
#                     Ensure you keep the headers with the '--': 
#                     -- Overview
#                     (your overview)
                  
#                     --Unique Value Proposition
#                     (your response)
                    
#                     --Competitors
#                     (your response)
                    
#                     -- Founding Story
#                     (your response)
                    
#                     --Mission & Vision
#                     (your response)

#                   --Values
#                     (your response)
#                     """})

#     draft_company_context = company_research['output']

#   # Save this generated version as the latest version
#     row['variable_value'] = draft_company_context
#     row.update()
#     print("Company Research Complete")
  
#     anvil.server.task_state['result'] = draft_company_context

  
# PRODUCT 1st DRAFT
@anvil.server.callable
def remove_duplicate_substrings(text, min_len=20):
    seen = OrderedDict()
    output_list = []
    n = len(text)

    for i in range(n - min_len + 1):
        substring = text[i:i + min_len]

        if substring not in seen:
            seen[substring] = i  # Save the starting index of this substring
            output_list.append(substring)

    # Join the substrings to get the final string
    return ''.join(output_list)

# HERE'S THE FUNCTION
@anvil.server.callable
def launch_draft_deepdive_product_1_generator(user_table,company_name,product_name,product_url):
    # Launch the background task

  # START THE WEB SCRAPING
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
    page_content = requests.get(product_url, headers=headers).content

    soup = BeautifulSoup(page_content, "html.parser")
    # Extract all the text from the page
    bulky_text_content = soup.get_text()
   # Remove leading and trailing whitespaces, replace newlines and extra spaces
    product_webpage_scraped_bulky = bulky_text_content.strip().replace('\n', ' ').replace('\r', '').replace('  ', ' ')

    # Further remove extra white spaces
    product_webpage_scraped = re.sub(r'\s+', ' ', product_webpage_scraped_bulky.strip())

    print("Scraped Information:",product_webpage_scraped)
  
    task = anvil.server.launch_background_task('deepdive_draft_product_1_generator',user_table,company_name,product_name,product_url,product_webpage_scraped)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def deepdive_draft_product_1_generator(user_table,company_name,product_name,product_url,product_webpage_scraped):
    print("Background task started for the Deep Dive of Researching the Product:", product_name)
    llm_agents = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo-16k', openai_api_key=openai_api_key)
    print("Background task started for generating the Product summary:", product_url)

    template_product_summary = """As a highly-skilled business research agent, your task is to conduct an exhaustive report and analysis of the company's product, {product_name} \
                  Leverage the product information that has been scraped from {company_name}'s' product website {product_url} in order to build your synopsis. However, note that there may be other products listed within the scraped information, so be diligent about your listed features. \
                  Lastly, be very specific! This is not an educational excercise. This work will be incorporated into our commercial operation shortly, so provide meaningful, actionable insights. Do not provide general terms or vague business ideas: be as particular about the issue as possible. Be confident. Provide numbers, statistics, prices, when possible!
                  \n \
                  Overview: Provide a comprehensive introduction to the product. What is its purpose, and what does the company aim to achieve with it? \n \
                  \n \
                  Description: Deeply describe the product. What does it look like, feel like, and what experience does it offer? \n \
                  \n \
                  Price: Detail the pricing structure. What is the cost, and are there any variations or tiers in pricing? \n \
                  \n \
                  Features: Elucidate the key features of the product. What distinguishes this product from others in the market? I would like around 15 differences between the product offers, if possible. \n \
                  \n \
                  Benefits: Explicate on how the product will benefit the customer. How can it change their life or improve their situation? \n \
                  \n \
                  Why people buy it: Analyze the consumer's pain points and desires before purchasing this product. Why might someone be drawn to this product, and what needs does it fulfill? \n \
                  \n \
                  Expected results: What are the anticipated outcomes or gains after using this product? How will the customer's situation improve or change? \n \
                  \n \
                  Guarantees: Discuss any guarantees the company offers with this product. Are there any assurances of product performance, or return policies in place? \n \
                  \n \
                  Bonuses: List any additional bonuses or incentives that come along with the product. What additional value does the company provide to sweeten the deal? \n \
                  \n \
                  Possible objections: Predict potential objections or concerns a customer may have about the product. How might the company address these? \n \
                  \n \
                  Ensure to provide an in-depth report with approximately 800-1000 words on the product, making it as detailed and specific as possible. Your aim is to capture the full essence of the product.
                  \n \
                  NOTES ON FORMAT:
                  Be confident. However, if there is incomplete information, please state "MORE INFORMATION NEEDED"! Speak as if you are the authority of the subject. 

                  ** END OF FORMAT
                  
                  FINALLY, HERE IS THE PRODUCT CONTEXT SCRAPED FROM THEIR PRODUCT WEBSITE: {product_webpage_scraped}
                  """
  
    prompt_product_summary = PromptTemplate(
        input_variables=["company_name", "product_name","product_url","product_webpage_scraped"],
        template=template_product_summary
    )
  
    chain_product_summary = LLMChain(llm=llm_agents, prompt=prompt_product_summary)
    product_summary = chain_product_summary.run(company_name=company_name,product_name=product_name,product_url=product_url,product_webpage_scraped=product_webpage_scraped)  # Pass in the combined context

   # Save it in the table:
    product_1_latest_row = user_table.search(variable='product_1_latest')[0]
    product_1_latest_row['variable_value'] = product_summary
    product_1_latest_row.update()
    print("Product Research Complete")
  
# PRODUCT 2, 1st DRAFT
@anvil.server.callable
def launch_draft_deepdive_product_2_generator(user_table,company_name,product_name,product_url):
    # Launch the background task

  # START THE WEB SCRAPING
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
    page_content = requests.get(product_url, headers=headers).content

    soup = BeautifulSoup(page_content, "html.parser")
    # Extract all the text from the page
    bulky_text_content = soup.get_text()
  # Remove leading and trailing whitespaces, replace newlines and extra spaces
    product_webpage_scraped_bulky = bulky_text_content.strip().replace('\n', ' ').replace('\r', '').replace('  ', ' ')

    # Further remove extra white spaces
    product_webpage_scraped = re.sub(r'\s+', ' ', product_webpage_scraped_bulky.strip())

    print("Scraped Information:",product_webpage_scraped)
  
    task = anvil.server.launch_background_task('deepdive_draft_product_2_generator',user_table,company_name,product_name,product_url,product_webpage_scraped)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def deepdive_draft_product_2_generator(user_table,company_name,product_name,product_url,product_webpage_scraped):
    print("Background task started for the Deep Dive of Researching the Product:", product_name)
    llm_agents = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo-16k', openai_api_key=openai_api_key)
    print("Background task started for generating the Product summary:", product_url)

    template_product_summary = """As a highly-skilled business research agent, your task is to conduct an exhaustive report and analysis of the company's product, {product_name} \
                  Leverage the product information that has been scraped from {company_name}'s' product website {product_url} in order to build your synopsis. However, note that there may be other products listed within the scraped information, so be diligent about your listed features. \
                  Lastly, be very specific! This is not an educational excercise. This work will be incorporated into our commercial operation shortly, so provide meaningful, actionable insights. Do not provide general terms or vague business ideas: be as particular about the issue as possible. Be confident. Provide numbers, statistics, prices, when possible!
                  \n \
                  Overview: Provide a comprehensive introduction to the product. What is its purpose, and what does the company aim to achieve with it? \n \
                  \n \
                  Description: Deeply describe the product. What does it look like, feel like, and what experience does it offer? \n \
                  \n \
                  Price: Detail the pricing structure. What is the cost, and are there any variations or tiers in pricing? \n \
                  \n \
                  Features: Elucidate the key features of the product. What distinguishes this product from others in the market? I would like around 15 differences between the product offers, if possible. \n \
                  \n \
                  Benefits: Explicate on how the product will benefit the customer. How can it change their life or improve their situation? \n \
                  \n \
                  Why people buy it: Analyze the consumer's pain points and desires before purchasing this product. Why might someone be drawn to this product, and what needs does it fulfill? \n \
                  \n \
                  Expected results: What are the anticipated outcomes or gains after using this product? How will the customer's situation improve or change? \n \
                  \n \
                  Guarantees: Discuss any guarantees the company offers with this product. Are there any assurances of product performance, or return policies in place? \n \
                  \n \
                  Bonuses: List any additional bonuses or incentives that come along with the product. What additional value does the company provide to sweeten the deal? \n \
                  \n \
                  Possible objections: Predict potential objections or concerns a customer may have about the product. How might the company address these? \n \
                  \n \
                  Ensure to provide an in-depth report with approximately 800-1000 words on the product, making it as detailed and specific as possible. Your aim is to capture the full essence of the product.
                  \n \
                  NOTES ON FORMAT:
                  Be confident. However, if there is incomplete information, please state "MORE INFORMATION NEEDED"! Speak as if you are the authority of the subject. 

                  ** END OF FORMAT
                  
                  FINALLY, HERE IS THE PRODUCT CONTEXT SCRAPED FROM THEIR PRODUCT WEBSITE: {product_webpage_scraped}
                  """
  
    prompt_product_summary = PromptTemplate(
        input_variables=["company_name", "product_name","product_url","product_webpage_scraped"],
        template=template_product_summary
    )
  
    chain_product_summary = LLMChain(llm=llm_agents, prompt=prompt_product_summary)
    product_summary = chain_product_summary.run(company_name=company_name,product_name=product_name,product_url=product_url,product_webpage_scraped=product_webpage_scraped)  # Pass in the combined context

   # Save it in the table:
    product_2_latest_row = user_table.search(variable='product_2_latest')[0]
    product_2_latest_row['variable_value'] = product_summary
    product_2_latest_row.update()
    print("Product Research Complete")

# PRODUCT 3, 1st DRAFT
@anvil.server.callable
def launch_draft_deepdive_product_3_generator(user_table,company_name,product_name,product_url):
    # Launch the background task

  # START THE WEB SCRAPING
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
    page_content = requests.get(product_url, headers=headers).content

    soup = BeautifulSoup(page_content, "html.parser")
    # Extract all the text from the page
    bulky_text_content = soup.get_text()
     # Remove leading and trailing whitespaces, replace newlines and extra spaces
    product_webpage_scraped_bulky = bulky_text_content.strip().replace('\n', ' ').replace('\r', '').replace('  ', ' ')

    # Further remove extra white spaces
    product_webpage_scraped = re.sub(r'\s+', ' ', product_webpage_scraped_bulky.strip())

    print("Scraped Information:",product_webpage_scraped)
  
    task = anvil.server.launch_background_task('deepdive_draft_product_3_generator',user_table,company_name,product_name,product_url,product_webpage_scraped)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def deepdive_draft_product_3_generator(user_table,company_name,product_name,product_url,product_webpage_scraped):
    print("Background task started for the Deep Dive of Researching the Product:", product_name)
    llm_agents = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo-16k', openai_api_key=openai_api_key)
    print("Background task started for generating the Product summary:", product_url)

    template_product_summary = """As a highly-skilled business research agent, your task is to conduct an exhaustive report and analysis of the company's product, {product_name} \
                  Leverage the product information that has been scraped from {company_name}'s' product website {product_url} in order to build your synopsis. However, note that there may be other products listed within the scraped information, so be diligent about your listed features. \
                  Lastly, be very specific! This is not an educational excercise. This work will be incorporated into our commercial operation shortly, so provide meaningful, actionable insights. Do not provide general terms or vague business ideas: be as particular about the issue as possible. Be confident. Provide numbers, statistics, prices, when possible!
                  \n \
                  Overview: Provide a comprehensive introduction to the product. What is its purpose, and what does the company aim to achieve with it? \n \
                  \n \
                  Description: Deeply describe the product. What does it look like, feel like, and what experience does it offer? \n \
                  \n \
                  Price: Detail the pricing structure. What is the cost, and are there any variations or tiers in pricing? \n \
                  \n \
                  Features: Elucidate the key features of the product. What distinguishes this product from others in the market? I would like around 15 differences between the product offers, if possible. \n \
                  \n \
                  Benefits: Explicate on how the product will benefit the customer. How can it change their life or improve their situation? \n \
                  \n \
                  Why people buy it: Analyze the consumer's pain points and desires before purchasing this product. Why might someone be drawn to this product, and what needs does it fulfill? \n \
                  \n \
                  Expected results: What are the anticipated outcomes or gains after using this product? How will the customer's situation improve or change? \n \
                  \n \
                  Guarantees: Discuss any guarantees the company offers with this product. Are there any assurances of product performance, or return policies in place? \n \
                  \n \
                  Bonuses: List any additional bonuses or incentives that come along with the product. What additional value does the company provide to sweeten the deal? \n \
                  \n \
                  Possible objections: Predict potential objections or concerns a customer may have about the product. How might the company address these? \n \
                  \n \
                  Ensure to provide an in-depth report with approximately 800-1000 words on the product, making it as detailed and specific as possible. Your aim is to capture the full essence of the product.
                  \n \
                  NOTES ON FORMAT:
                  Be confident. However, if there is incomplete information, please state "MORE INFORMATION NEEDED"! Speak as if you are the authority of the subject. 

                  ** END OF FORMAT
                  
                  FINALLY, HERE IS THE PRODUCT CONTEXT SCRAPED FROM THEIR PRODUCTWEBSITE: {product_webpage_scraped}
                  """
  
    prompt_product_summary = PromptTemplate(
        input_variables=["company_name", "product_name","product_url","product_webpage_scraped"],
        template=template_product_summary
    )
  
    chain_product_summary = LLMChain(llm=llm_agents, prompt=prompt_product_summary)
    product_summary = chain_product_summary.run(company_name=company_name,product_name=product_name,product_url=product_url,product_webpage_scraped=product_webpage_scraped)  # Pass in the combined context

   # Save it in the table:
    product_3_latest_row = user_table.search(variable='product_3_latest')[0]
    product_3_latest_row['variable_value'] = product_summary
    product_3_latest_row.update()
    print("Product Research Complete")

# PRODUCT 4, 1st DRAFT
@anvil.server.callable
def launch_draft_deepdive_product_4_generator(user_table,company_name,product_name,product_url):
    # Launch the background task

  # START THE WEB SCRAPING
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
    page_content = requests.get(product_url, headers=headers).content

    soup = BeautifulSoup(page_content, "html.parser")
    # Extract all the text from the page
    bulky_text_content = soup.get_text()
     # Remove leading and trailing whitespaces, replace newlines and extra spaces
    product_webpage_scraped_bulky = bulky_text_content.strip().replace('\n', ' ').replace('\r', '').replace('  ', ' ')

    # Further remove extra white spaces
    product_webpage_scraped = re.sub(r'\s+', ' ', product_webpage_scraped_bulky.strip())

    print("Scraped Information:",product_webpage_scraped)
  
    task = anvil.server.launch_background_task('deepdive_draft_product_4_generator',user_table,company_name,product_name,product_url,product_webpage_scraped)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def deepdive_draft_product_4_generator(user_table,company_name,product_name,product_url,product_webpage_scraped):
    print("Background task started for the Deep Dive of Researching the Product:", product_name)
    llm_agents = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo-16k', openai_api_key=openai_api_key)
    print("Background task started for generating the Product summary:", product_url)

    template_product_summary = """As a highly-skilled business research agent, your task is to conduct an exhaustive report and analysis of the company's product, {product_name} \
                  Leverage the product information that has been scraped from {company_name}'s' product website {product_url} in order to build your synopsis. However, note that there may be other products listed within the scraped information, so be diligent about your listed features. \
                  Lastly, be very specific! This is not an educational excercise. This work will be incorporated into our commercial operation shortly, so provide meaningful, actionable insights. Do not provide general terms or vague business ideas: be as particular about the issue as possible. Be confident. Provide numbers, statistics, prices, when possible!
                  \n \
                  Overview: Provide a comprehensive introduction to the product. What is its purpose, and what does the company aim to achieve with it? \n \
                  \n \
                  Description: Deeply describe the product. What does it look like, feel like, and what experience does it offer? \n \
                  \n \
                  Price: Detail the pricing structure. What is the cost, and are there any variations or tiers in pricing? \n \
                  \n \
                  Features: Elucidate the key features of the product. What distinguishes this product from others in the market? I would like around 15 differences between the product offers, if possible. \n \
                  \n \
                  Benefits: Explicate on how the product will benefit the customer. How can it change their life or improve their situation? \n \
                  \n \
                  Why people buy it: Analyze the consumer's pain points and desires before purchasing this product. Why might someone be drawn to this product, and what needs does it fulfill? \n \
                  \n \
                  Expected results: What are the anticipated outcomes or gains after using this product? How will the customer's situation improve or change? \n \
                  \n \
                  Guarantees: Discuss any guarantees the company offers with this product. Are there any assurances of product performance, or return policies in place? \n \
                  \n \
                  Bonuses: List any additional bonuses or incentives that come along with the product. What additional value does the company provide to sweeten the deal? \n \
                  \n \
                  Possible objections: Predict potential objections or concerns a customer may have about the product. How might the company address these? \n \
                  \n \
                  Ensure to provide an in-depth report with approximately 800-1000 words on the product, making it as detailed and specific as possible. Your aim is to capture the full essence of the product.
                  \n \
                  NOTES ON FORMAT:
                  Be confident. However, if there is incomplete information, please state "MORE INFORMATION NEEDED"! Speak as if you are the authority of the subject. 

                  ** END OF FORMAT
                  
                  FINALLY, HERE IS THE PRODUCT CONTEXT SCRAPED FROM THEIR PRODUCT WEBSITE: {product_webpage_scraped}
                  """
  
    prompt_product_summary = PromptTemplate(
        input_variables=["company_name", "product_name","product_url","product_webpage_scraped"],
        template=template_product_summary
    )
  
    chain_product_summary = LLMChain(llm=llm_agents, prompt=prompt_product_summary)
    product_summary = chain_product_summary.run(company_name=company_name,product_name=product_name,product_url=product_url,product_webpage_scraped=product_webpage_scraped)  # Pass in the combined context

   # Save it in the table:
    product_4_latest_row = user_table.search(variable='product_4_latest')[0]
    product_4_latest_row['variable_value'] = product_summary
    product_4_latest_row.update()
    print("Product Research Complete")

# PRODUCT 5, 1st DRAFT
@anvil.server.callable
def launch_draft_deepdive_product_5_generator(user_table,company_name,product_name,product_url):
    # Launch the background task

  # START THE WEB SCRAPING
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
    page_content = requests.get(product_url, headers=headers).content

    soup = BeautifulSoup(page_content, "html.parser")
    # Extract all the text from the page
    bulky_text_content = soup.get_text()
    # Remove leading and trailing whitespaces, replace newlines and extra spaces
    product_webpage_scraped_bulky = bulky_text_content.strip().replace('\n', ' ').replace('\r', '').replace('  ', ' ')

    # Further remove extra white spaces
    product_webpage_scraped = re.sub(r'\s+', ' ', product_webpage_scraped_bulky.strip())

    print("Scraped Information:",product_webpage_scraped)
  
    task = anvil.server.launch_background_task('deepdive_draft_product_5_generator',user_table,company_name,product_name,product_url,product_webpage_scraped)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def deepdive_draft_product_5_generator(user_table,company_name,product_name,product_url,product_webpage_scraped):
    print("Background task started for the Deep Dive of Researching the Product:", product_name)
    llm_agents = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo-16k', openai_api_key=openai_api_key)
    print("Background task started for generating the Product summary:", product_url)

    template_product_summary = """As a highly-skilled business research agent, your task is to conduct an exhaustive report and analysis of the company's product, {product_name} \
                  Leverage the product information that has been scraped from {company_name}'s' product website {product_url} in order to build your synopsis. However, note that there may be other products listed within the scraped information, so be diligent about your listed features. \
                  Lastly, be very specific! This is not an educational excercise. This work will be incorporated into our commercial operation shortly, so provide meaningful, actionable insights. Do not provide general terms or vague business ideas: be as particular about the issue as possible. Be confident. Provide numbers, statistics, prices, when possible!
                  \n \
                  Overview: Provide a comprehensive introduction to the product. What is its purpose, and what does the company aim to achieve with it? \n \
                  \n \
                  Description: Deeply describe the product. What does it look like, feel like, and what experience does it offer? \n \
                  \n \
                  Price: Detail the pricing structure. What is the cost, and are there any variations or tiers in pricing? \n \
                  \n \
                  Features: Elucidate the key features of the product. What distinguishes this product from others in the market? I would like around 15 differences between the product offers, if possible. \n \
                  \n \
                  Benefits: Explicate on how the product will benefit the customer. How can it change their life or improve their situation? \n \
                  \n \
                  Why people buy it: Analyze the consumer's pain points and desires before purchasing this product. Why might someone be drawn to this product, and what needs does it fulfill? \n \
                  \n \
                  Expected results: What are the anticipated outcomes or gains after using this product? How will the customer's situation improve or change? \n \
                  \n \
                  Guarantees: Discuss any guarantees the company offers with this product. Are there any assurances of product performance, or return policies in place? \n \
                  \n \
                  Bonuses: List any additional bonuses or incentives that come along with the product. What additional value does the company provide to sweeten the deal? \n \
                  \n \
                  Possible objections: Predict potential objections or concerns a customer may have about the product. How might the company address these? \n \
                  \n \
                  Ensure to provide an in-depth report with approximately 800-1000 words on the product, making it as detailed and specific as possible. Your aim is to capture the full essence of the product.
                  \n \
                  NOTES ON FORMAT:
                  Be confident. However, if there is incomplete information, please state "MORE INFORMATION NEEDED"! Speak as if you are the authority of the subject. 

                  ** END OF FORMAT
                  
                  FINALLY, HERE IS THE PRODUCT CONTEXT SCRAPED FROM THEIR PRODUCT WEBSITE: {product_webpage_scraped}
                  """
  
    prompt_product_summary = PromptTemplate(
        input_variables=["company_name", "product_name","product_url","product_webpage_scraped"],
        template=template_product_summary
    )
  
    chain_product_summary = LLMChain(llm=llm_agents, prompt=prompt_product_summary)
    product_summary = chain_product_summary.run(company_name=company_name,product_name=product_name,product_url=product_url,product_webpage_scraped=product_webpage_scraped)  # Pass in the combined context

   # Save it in the table:
    product_5_latest_row = user_table.search(variable='product_5_latest')[0]
    product_5_latest_row['variable_value'] = product_summary
    product_5_latest_row.update()
    print("Product Research Complete")
  
#------AVATARS, 1st DRAFT - AVATAR 1 / PRODUCT 1-------------------------------------------##################
@anvil.server.callable
def launch_draft_deepdive_avatar_1_product_1_generator(user_table,company_name,product_1_name,avatar_1_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('draft_deepdive_avatar_1_product_1_generator', user_table,company_name,product_1_name,avatar_1_preview)
    # Return the task ID
    return task.get_id()
@anvil.server.background_task
def draft_deepdive_avatar_1_product_1_generator(user_table,company_name,product_1_name,avatar_1_preview):
    print("Background task started for generating the avatar:", avatar_1_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company, {company_name}, who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would build a customer avatar. Please prepare the ideal customer avatar, that is, the ideal 'dream' customer who would purchase the below product or service. 

    Company Context: The company, {company_name}, is selling {product_1_name}.
    Here's a quick snapshot of the description of this ideal customer avatar you are to expand on to develop into a detailed avatar: {avatar_1_preview}
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
   
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["company_name","product_1_name", "avatar_1_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    draft_avatar = chain_avatar.run(company_name=company_name, product_1_name=product_1_name, avatar_1_preview=avatar_1_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = draft_avatar

    # Save this generated version as the latest version
    row_avatar_1_latest = user_table.search(variable='avatar_1_product_1_latest')
    first_row_avatar_1_latest = row_avatar_1_latest[0]
    first_row_avatar_1_latest['variable_value'] = draft_avatar
    first_row_avatar_1_latest.update()
    print("Avatar Draft Research Complete")

# AVATARS, 1st DRAFT - AVATAR 2 / PRODUCT 1
@anvil.server.callable
def launch_draft_deepdive_avatar_2_product_1_generator(user_table,company_name,product_1_name,avatar_2_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('draft_deepdive_avatar_2_product_1_generator', user_table,company_name,product_1_name,avatar_2_preview)
    # Return the task ID
    return task.get_id()  
@anvil.server.background_task
def draft_deepdive_avatar_2_product_1_generator(user_table,company_name,product_1_name,avatar_2_preview):
    print("Background task started for generating the avatar:", avatar_2_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company, {company_name}, who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would build a customer avatar. Please prepare the ideal customer avatar, that is, the ideal 'dream' customer who would purchase the below product or service. 

    Company Context: The company, {company_name}, is selling {product_1_name}.
    Here's a quick snapshot of the description of this ideal customer avatar you are to expand on to develop into a detailed avatar: {avatar_2_preview}
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
   
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["company_name","product_1_name", "avatar_2_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    draft_avatar = chain_avatar.run(company_name=company_name, product_1_name=product_1_name, avatar_2_preview=avatar_2_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = draft_avatar

    # Save this generated version as the latest version
    row_avatar_2_latest = user_table.search(variable='avatar_2_product_1_latest')
    first_row_avatar_2_latest = row_avatar_2_latest[0]
    first_row_avatar_2_latest['variable_value'] = draft_avatar
    first_row_avatar_2_latest.update()
    print("Avatar Draft Research Complete")

# AVATARS, 1st DRAFT - AVATAR 3 / PRODUCT 1
@anvil.server.callable
def launch_draft_deepdive_avatar_3_product_1_generator(user_table,company_name,product_1_name,avatar_3_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('draft_deepdive_avatar_3_product_1_generator', user_table,company_name,product_1_name,avatar_3_preview)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def draft_deepdive_avatar_3_product_1_generator(user_table,company_name,product_1_name,avatar_3_preview):
    print("Background task started for generating the avatar:", avatar_3_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company, {company_name}, who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would build a customer avatar. Please prepare the ideal customer avatar, that is, the ideal 'dream' customer who would purchase the below product or service. 

    Company Context: The company, {company_name}, is selling {product_1_name}.
    Here's a quick snapshot of the description of this ideal customer avatar you are to expand on to develop into a detailed avatar: {avatar_3_preview}
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
   
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["company_name","product_1_name", "avatar_3_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    draft_avatar = chain_avatar.run(company_name=company_name, product_1_name=product_1_name, avatar_3_preview=avatar_3_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = draft_avatar

    # Save this generated version as the latest version
    row_avatar_3_latest = user_table.search(variable='avatar_3_product_1_latest')
    first_row_avatar_3_latest = row_avatar_3_latest[0]
    first_row_avatar_3_latest['variable_value'] = draft_avatar
    first_row_avatar_3_latest.update()
    print("Avatar Draft Research Complete")

#------AVATARS, 1st DRAFT - AVATAR 1 / PRODUCT 2-----------------##################
@anvil.server.callable
def launch_draft_deepdive_avatar_1_product_2_generator(user_table,company_name,product_2_name,avatar_1_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('draft_deepdive_avatar_1_product_2_generator', user_table,company_name,product_2_name,avatar_1_preview)
    # Return the task ID
    return task.get_id()
@anvil.server.background_task
def draft_deepdive_avatar_1_product_2_generator(user_table,company_name,product_2_name,avatar_1_preview):
    print("Background task started for generating the avatar:", avatar_1_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company, {company_name}, who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would build a customer avatar. Please prepare the ideal customer avatar, that is, the ideal 'dream' customer who would purchase the below product or service. 

    Company Context: The company, {company_name}, is selling {product_2_name}.
    Here's a quick snapshot of the description of this ideal customer avatar you are to expand on to develop into a detailed avatar: {avatar_1_preview}
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
   
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["company_name","product_2_name", "avatar_1_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    draft_avatar = chain_avatar.run(company_name=company_name, product_2_name=product_2_name, avatar_1_preview=avatar_1_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = draft_avatar

    # Save this generated version as the latest version
    row_avatar_1_latest = user_table.search(variable='avatar_1_product_2_latest')
    first_row_avatar_1_latest = row_avatar_1_latest[0]
    first_row_avatar_1_latest['variable_value'] = draft_avatar
    first_row_avatar_1_latest.update()
    print("Avatar Draft Research Complete")

# AVATARS, 1st DRAFT - AVATAR 2 / PRODUCT 2
@anvil.server.callable
def launch_draft_deepdive_avatar_2_product_2_generator(user_table,company_name,product_2_name,avatar_2_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('draft_deepdive_avatar_2_product_2_generator', user_table,company_name,product_2_name,avatar_2_preview)
    # Return the task ID
    return task.get_id()  
@anvil.server.background_task
def draft_deepdive_avatar_2_product_2_generator(user_table,company_name,product_2_name,avatar_2_preview):
    print("Background task started for generating the avatar:", avatar_2_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company, {company_name}, who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would build a customer avatar. Please prepare the ideal customer avatar, that is, the ideal 'dream' customer who would purchase the below product or service. 

    Company Context: The company, {company_name}, is selling {product_2_name}.
    Here's a quick snapshot of the description of this ideal customer avatar you are to expand on to develop into a detailed avatar: {avatar_2_preview}
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
   
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["company_name","product_2_name", "avatar_2_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    draft_avatar = chain_avatar.run(company_name=company_name, product_2_name=product_2_name, avatar_2_preview=avatar_2_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = draft_avatar

    # Save this generated version as the latest version
    row_avatar_2_latest = user_table.search(variable='avatar_2_product_2_latest')
    first_row_avatar_2_latest = row_avatar_2_latest[0]
    first_row_avatar_2_latest['variable_value'] = draft_avatar
    first_row_avatar_2_latest.update()
    print("Avatar Draft Research Complete")

# AVATARS, 1st DRAFT - AVATAR 3 / PRODUCT 2
@anvil.server.callable
def launch_draft_deepdive_avatar_3_product_2_generator(user_table,company_name,product_2_name,avatar_3_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('draft_deepdive_avatar_3_product_2_generator', user_table,company_name,product_2_name,avatar_3_preview)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def draft_deepdive_avatar_3_product_2_generator(user_table,company_name,product_2_name,avatar_3_preview):
    print("Background task started for generating the avatar:", avatar_3_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company, {company_name}, who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would build a customer avatar. Please prepare the ideal customer avatar, that is, the ideal 'dream' customer who would purchase the below product or service. 

    Company Context: The company, {company_name}, is selling {product_2_name}.
    Here's a quick snapshot of the description of this ideal customer avatar you are to expand on to develop into a detailed avatar: {avatar_3_preview}
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
   
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["company_name","product_2_name", "avatar_3_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    draft_avatar = chain_avatar.run(company_name=company_name, product_2_name=product_2_name, avatar_3_preview=avatar_3_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = draft_avatar

    # Save this generated version as the latest version
    row_avatar_3_latest = user_table.search(variable='avatar_3_product_2_latest')
    first_row_avatar_3_latest = row_avatar_3_latest[0]
    first_row_avatar_3_latest['variable_value'] = draft_avatar
    first_row_avatar_3_latest.update()
    print("Avatar Draft Research Complete")

#------AVATARS, 1st DRAFT - AVATAR 1 / PRODUCT 3 -----------------##################
@anvil.server.callable
def launch_draft_deepdive_avatar_1_product_3_generator(user_table,company_name,product_3_name,avatar_1_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('draft_deepdive_avatar_1_product_3_generator', user_table,company_name,product_3_name,avatar_1_preview)
    # Return the task ID
    return task.get_id()
@anvil.server.background_task
def draft_deepdive_avatar_1_product_3_generator(user_table,company_name,product_3_name,avatar_1_preview):
    print("Background task started for generating the avatar:", avatar_1_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company, {company_name}, who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would build a customer avatar. Please prepare the ideal customer avatar, that is, the ideal 'dream' customer who would purchase the below product or service. 

    Company Context: The company, {company_name}, is selling {product_3_name}.
    Here's a quick snapshot of the description of this ideal customer avatar you are to expand on to develop into a detailed avatar: {avatar_1_preview}
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
   
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["company_name","product_3_name", "avatar_1_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    draft_avatar = chain_avatar.run(company_name=company_name, product_3_name=product_3_name, avatar_1_preview=avatar_1_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = draft_avatar

    # Save this generated version as the latest version
    row_avatar_1_latest = user_table.search(variable='avatar_1_product_3_latest')
    first_row_avatar_1_latest = row_avatar_1_latest[0]
    first_row_avatar_1_latest['variable_value'] = draft_avatar
    first_row_avatar_1_latest.update()
    print("Avatar Draft Research Complete")

# AVATARS, 1st DRAFT - AVATAR 2 / PRODUCT 3
@anvil.server.callable
def launch_draft_deepdive_avatar_2_product_3_generator(user_table,company_name,product_3_name,avatar_2_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('draft_deepdive_avatar_2_product_3_generator', user_table,company_name,product_3_name,avatar_2_preview)
    # Return the task ID
    return task.get_id()  
@anvil.server.background_task
def draft_deepdive_avatar_2_product_3_generator(user_table,company_name,product_3_name,avatar_2_preview):
    print("Background task started for generating the avatar:", avatar_2_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company, {company_name}, who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would build a customer avatar. Please prepare the ideal customer avatar, that is, the ideal 'dream' customer who would purchase the below product or service. 

    Company Context: The company, {company_name}, is selling {product_3_name}.
    Here's a quick snapshot of the description of this ideal customer avatar you are to expand on to develop into a detailed avatar: {avatar_2_preview}
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
   
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["company_name","product_3_name", "avatar_2_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    draft_avatar = chain_avatar.run(company_name=company_name, product_3_name=product_3_name, avatar_2_preview=avatar_2_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = draft_avatar

    # Save this generated version as the latest version
    row_avatar_2_latest = user_table.search(variable='avatar_2_product_3_latest')
    first_row_avatar_2_latest = row_avatar_2_latest[0]
    first_row_avatar_2_latest['variable_value'] = draft_avatar
    first_row_avatar_2_latest.update()
    print("Avatar Draft Research Complete")

# AVATARS, 1st DRAFT - AVATAR 3 / PRODUCT 3
@anvil.server.callable
def launch_draft_deepdive_avatar_3_product_3_generator(user_table,company_name,product_3_name,avatar_3_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('draft_deepdive_avatar_3_product_3_generator', user_table,company_name,product_3_name,avatar_3_preview)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def draft_deepdive_avatar_3_product_3_generator(user_table,company_name,product_3_name,avatar_3_preview):
    print("Background task started for generating the avatar:", avatar_3_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company, {company_name}, who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would build a customer avatar. Please prepare the ideal customer avatar, that is, the ideal 'dream' customer who would purchase the below product or service. 

    Company Context: The company, {company_name}, is selling {product_3_name}.
    Here's a quick snapshot of the description of this ideal customer avatar you are to expand on to develop into a detailed avatar: {avatar_3_preview}
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
   
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["company_name","product_3_name", "avatar_3_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    draft_avatar = chain_avatar.run(company_name=company_name, product_3_name=product_3_name, avatar_3_preview=avatar_3_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = draft_avatar

    # Save this generated version as the latest version
    row_avatar_3_latest = user_table.search(variable='avatar_3_product_3_latest')
    first_row_avatar_3_latest = row_avatar_3_latest[0]
    first_row_avatar_3_latest['variable_value'] = draft_avatar
    first_row_avatar_3_latest.update()
    print("Avatar Draft Research Complete")

# AVATARS, 1st DRAFT - AVATAR 1 / PRODUCT 4
@anvil.server.callable
def launch_draft_deepdive_avatar_1_product_4_generator(user_table,company_name,product_4_name,avatar_1_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('draft_deepdive_avatar_1_product_4_generator', user_table,company_name,product_4_name,avatar_1_preview)
    # Return the task ID
    return task.get_id()
@anvil.server.background_task
def draft_deepdive_avatar_1_product_4_generator(user_table,company_name,product_4_name,avatar_1_preview):
    print("Background task started for generating the avatar:", avatar_1_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company, {company_name}, who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would build a customer avatar. Please prepare the ideal customer avatar, that is, the ideal 'dream' customer who would purchase the below product or service. 

    Company Context: The company, {company_name}, is selling {product_4_name}.
    Here's a quick snapshot of the description of this ideal customer avatar you are to expand on to develop into a detailed avatar: {avatar_1_preview}
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
   
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["company_name","product_4_name", "avatar_1_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    draft_avatar = chain_avatar.run(company_name=company_name, product_4_name=product_4_name, avatar_1_preview=avatar_1_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = draft_avatar

    # Save this generated version as the latest version
    row_avatar_1_latest = user_table.search(variable='avatar_1_product_4_latest')
    first_row_avatar_1_latest = row_avatar_1_latest[0]
    first_row_avatar_1_latest['variable_value'] = draft_avatar
    first_row_avatar_1_latest.update()
    print("Avatar Draft Research Complete")

# AVATARS, 1st DRAFT - AVATAR 2 / PRODUCT 4
@anvil.server.callable
def launch_draft_deepdive_avatar_2_product_4_generator(user_table,company_name,product_4_name,avatar_2_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('draft_deepdive_avatar_2_product_4_generator', user_table,company_name,product_4_name,avatar_2_preview)
    # Return the task ID
    return task.get_id()  
@anvil.server.background_task
def draft_deepdive_avatar_2_product_4_generator(user_table,company_name,product_4_name,avatar_2_preview):
    print("Background task started for generating the avatar:", avatar_2_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company, {company_name}, who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would build a customer avatar. Please prepare the ideal customer avatar, that is, the ideal 'dream' customer who would purchase the below product or service. 

    Company Context: The company, {company_name}, is selling {product_4_name}.
    Here's a quick snapshot of the description of this ideal customer avatar you are to expand on to develop into a detailed avatar: {avatar_2_preview}
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
   
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["company_name","product_4_name", "avatar_2_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    draft_avatar = chain_avatar.run(company_name=company_name, product_4_name=product_4_name, avatar_2_preview=avatar_2_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = draft_avatar

    # Save this generated version as the latest version
    row_avatar_2_latest = user_table.search(variable='avatar_2_product_4_latest')
    first_row_avatar_2_latest = row_avatar_2_latest[0]
    first_row_avatar_2_latest['variable_value'] = draft_avatar
    first_row_avatar_2_latest.update()
    print("Avatar Draft Research Complete")

# AVATARS, 1st DRAFT - AVATAR 3 / PRODUCT 4
@anvil.server.callable
def launch_draft_deepdive_avatar_3_product_4_generator(user_table,company_name,product_4_name,avatar_3_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('draft_deepdive_avatar_3_product_4_generator', user_table,company_name,product_4_name,avatar_3_preview)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def draft_deepdive_avatar_3_product_4_generator(user_table,company_name,product_4_name,avatar_3_preview):
    print("Background task started for generating the avatar:", avatar_3_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company, {company_name}, who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would build a customer avatar. Please prepare the ideal customer avatar, that is, the ideal 'dream' customer who would purchase the below product or service. 

    Company Context: The company, {company_name}, is selling {product_4_name}.
    Here's a quick snapshot of the description of this ideal customer avatar you are to expand on to develop into a detailed avatar: {avatar_3_preview}
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
   
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["company_name","product_4_name", "avatar_3_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    draft_avatar = chain_avatar.run(company_name=company_name, product_4_name=product_4_name, avatar_3_preview=avatar_3_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = draft_avatar

    # Save this generated version as the latest version
    row_avatar_3_latest = user_table.search(variable='avatar_3_product_4_latest')
    first_row_avatar_3_latest = row_avatar_3_latest[0]
    first_row_avatar_3_latest['variable_value'] = draft_avatar
    first_row_avatar_3_latest.update()
    print("Avatar Draft Research Complete")

# AVATARS, 1st DRAFT - AVATAR 1 / PRODUCT 5
@anvil.server.callable
def launch_draft_deepdive_avatar_1_product_5_generator(user_table,company_name,product_5_name,avatar_1_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('draft_deepdive_avatar_1_product_5_generator', user_table,company_name,product_5_name,avatar_1_preview)
    # Return the task ID
    return task.get_id()
@anvil.server.background_task
def draft_deepdive_avatar_1_product_5_generator(user_table,company_name,product_5_name,avatar_1_preview):
    print("Background task started for generating the avatar:", avatar_1_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company, {company_name}, who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would build a customer avatar. Please prepare the ideal customer avatar, that is, the ideal 'dream' customer who would purchase the below product or service. 

    Company Context: The company, {company_name}, is selling {product_5_name}.
    Here's a quick snapshot of the description of this ideal customer avatar you are to expand on to develop into a detailed avatar: {avatar_1_preview}
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
   
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["company_name","product_5_name", "avatar_1_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    draft_avatar = chain_avatar.run(company_name=company_name, product_5_name=product_5_name, avatar_1_preview=avatar_1_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = draft_avatar

    # Save this generated version as the latest version
    row_avatar_1_latest = user_table.search(variable='avatar_1_product_5_latest')
    first_row_avatar_1_latest = row_avatar_1_latest[0]
    first_row_avatar_1_latest['variable_value'] = draft_avatar
    first_row_avatar_1_latest.update()
    print("Avatar Draft Research Complete")

# AVATARS, 1st DRAFT - AVATAR 2 / PRODUCT 5
@anvil.server.callable
def launch_draft_deepdive_avatar_2_product_5_generator(user_table,company_name,product_5_name,avatar_2_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('draft_deepdive_avatar_2_product_5_generator', user_table,company_name,product_5_name,avatar_2_preview)
    # Return the task ID
    return task.get_id()  
@anvil.server.background_task
def draft_deepdive_avatar_2_product_5_generator(user_table,company_name,product_5_name,avatar_2_preview):
    print("Background task started for generating the avatar:", avatar_2_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company, {company_name}, who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would build a customer avatar. Please prepare the ideal customer avatar, that is, the ideal 'dream' customer who would purchase the below product or service. 

    Company Context: The company, {company_name}, is selling {product_5_name}.
    Here's a quick snapshot of the description of this ideal customer avatar you are to expand on to develop into a detailed avatar: {avatar_2_preview}
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
   
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["company_name","product_5_name", "avatar_2_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    draft_avatar = chain_avatar.run(company_name=company_name, product_5_name=product_5_name, avatar_2_preview=avatar_2_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = draft_avatar

    # Save this generated version as the latest version
    row_avatar_2_latest = user_table.search(variable='avatar_2_product_5_latest')
    first_row_avatar_2_latest = row_avatar_2_latest[0]
    first_row_avatar_2_latest['variable_value'] = draft_avatar
    first_row_avatar_2_latest.update()
    print("Avatar Draft Research Complete")

# AVATARS, 1st DRAFT - AVATAR 3 / PRODUCT 5
@anvil.server.callable
def launch_draft_deepdive_avatar_3_product_5_generator(user_table,company_name,product_5_name,avatar_3_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('draft_deepdive_avatar_3_product_5_generator', user_table,company_name,product_5_name,avatar_3_preview)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def draft_deepdive_avatar_3_product_5_generator(user_table,company_name,product_5_name,avatar_3_preview):
    print("Background task started for generating the avatar:", avatar_3_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company, {company_name}, who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would build a customer avatar. Please prepare the ideal customer avatar, that is, the ideal 'dream' customer who would purchase the below product or service. 

    Company Context: The company, {company_name}, is selling {product_5_name}.
    Here's a quick snapshot of the description of this ideal customer avatar you are to expand on to develop into a detailed avatar: {avatar_3_preview}
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
   
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["company_name","product_5_name", "avatar_3_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    draft_avatar = chain_avatar.run(company_name=company_name, product_5_name=product_5_name, avatar_3_preview=avatar_3_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = draft_avatar

    # Save this generated version as the latest version
    row_avatar_3_latest = user_table.search(variable='avatar_3_product_5_latest')
    first_row_avatar_3_latest = row_avatar_3_latest[0]
    first_row_avatar_3_latest['variable_value'] = draft_avatar
    first_row_avatar_3_latest.update()
    print("Avatar Draft Research Complete")
  
# BRAND TONE 1st DRAFT 
@anvil.server.callable
def launch_draft_brand_tone_research(user_table,company_url):
    # Launch the background task
    task = anvil.server.launch_background_task('draft_brand_tone_research',user_table,company_url)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def draft_brand_tone_research(user_table,brand_tone_url):
    print("Background task started for extracting brand tone:", user_table,brand_tone_url)
 
    llm_agents = ChatOpenAI(temperature=0.2, model_name='gpt-4', openai_api_key=openai_api_key)
    agent_tone_extraction = initialize_agent([tools], llm_agents, agent="zero-shot-react-description", handle_parsing_errors=True)
 
    tone_research = agent_tone_extraction({
      "input":f"""You are CopywriterAI, the best copywriter on the planet. We are looking to generate a description 
      of the tone that best describes the copywriting style and tone of an existing web page. Go and research this URL 
      {brand_tone_url}, and provide your analysis.

      For example, for PROFESSIONAL / ENTERPRISE, it would be described as:
        - 'Formal and Polished': (sophisticated language and complex sentence structures).
        - 'Objective and Analytical': (incorporates data and facts, prioritizing logical arguments).
        - 'Business-like': Efficient, frequently employs industry-specific jargon and business terminology).
        - 'Trustworthy and Reliable': Underscoring credibility, reliability, and accuracy.
        - 'Instructional and Informative': (providing clear, direct instructions or information).
        - 'Respectful and Considerate': (acknowledging the audience's needs and viewpoints while avoiding excessive casualness).
        - 'Controlled and Consistent': (providing coherence, ensuring careful, consistent writing).

        For Russell Brunson, sales page style, it would be described as:
        - 'Conversational': (friendly, casual, and approachable).
        - 'Storytelling': (using compelling stories to illustrate his points).
        - 'Educational': (being informative, teaching something new).
        - 'Persuasive': (being compelling and enticing, using ideas of scarcity (limited time offers), social proof (testimonials), and authority (expertise and success).
        - 'Inspiring': (motivating and inspiring, encouraging the reader to take action).
        - 'Clear and Direct': (providing clarity and simplicity, avoiding jargon).

        However, it is up to you to go and review the website, think about the tone of the existing copy, and return 5-6 descriptors, in the similar format as above. They don't have to be listed above- they can be new!
        
        OUTPUT TEMPLATE: AN EXAMPLE OUTPUT SHOULD BE AS BELOW:
        'The businesstone can be described as': 
        - 'Conversational': (friendly, casual, and approachable).
        - 'Storytelling': (using compelling stories to illustrate his points).
        - 'Educational': (being informative, teaching something new).
        - 'Persuasive': (being compelling and enticing, using ideas of scarcity (limited time offers), social proof (testimonials), and authority (expertise and success).
        - 'Inspiring': (motivating and inspiring, encouraging the reader to take action).
        - 'Clear and Direct': (providing clarity and simplicity, avoiding jargon).Conversational, Storytelling, Educational, Persuasive, Inspiring, Clear and Direct'
        
        FINAL RULES: Don't mention the business name, or source. Just say "the business" and refer to it as 'company tone' or 'the business tone'
        """})

    extracted_tone = tone_research['output']
    anvil.server.task_state['result'] = extracted_tone
  
    # Save the brand tone 
    brand_tone_latest_row = list(user_table.search(variable='brand_tone'))
    first_row_brand_tone_latest =  brand_tone_latest_row[0]
    first_row_brand_tone_latest['variable_value'] = extracted_tone
    first_row_brand_tone_latest['variable_title'] = brand_tone_url
    first_row_brand_tone_latest.update()
    print("Brand Tone Research Complete")

# Function to get the status of a background task
@anvil.server.callable
def get_status_function(task_id):
    # Retrieve the task status from the Data Table (assuming you have a Data Table named 'tasks')
    task_table = app_tables.tasks  # Replace 'tasks' with your actual Data Table name
    task_row = task_table.get(task_id=task_id)
    status = task_row['status']
    return status
  
####### -------------------------------- COMPANY ----------------------------------------------------###########
@anvil.server.callable
def launch_company_summary(company_name, company_url):
    # Launch the background task
    print("Launch task started for researching company:",company_name,company_url)
    task = anvil.server.launch_background_task('company_summary', company_name, company_url)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def company_summary(company_name, company_url):
    print("Background task started for researching company:", company_name,company_url)
    # Here, you should write the code that uses the company_name and company_url
    # to research the company and generate a context. For example:
  
    llm_agents = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo-16k', openai_api_key=openai_api_key)
    agent_company_context = initialize_agent([tools], llm_agents, agent="zero-shot-react-description", handle_parsing_errors=True) #max_execution_time=300,max_iterations=300
    company_research = agent_company_context({"input": f"""As a highly-skilled business research agent, your task is to conduct an exhaustive analysis to build an informational company profile of {company_name}. \
                    Leverage all necessary resources, primarily the company's website {company_url}, but also news articles, and any other relevant sources.  \
                    to gather the following details about {company_name}. Lastly, be very specific! This is not an educational excercise. This work will be incorporated into our commercial operation shortly, so provide meaningful research and findings. Do not provide general terms or vague business ideas: be as particular about the issue as possible. Be confident. Provide numbers, statistics, prices, when possible!
                    \n \
                    Overview: Provide a comprehensive introduction to the company. What are the unique features or value propositions of the company's offerings? What does the company aim to achieve? \n \
                    \n \
                     Unique Value Proposition: What is the company unique value proposition? What are they uniquely positioned to do? How does their main offer differ from their competitors? \n \
                    \n \
                    Founding Story: What inspired the founders to start the company? Are there any unique or interesting anecdotes about the early days of the company? How has the company evolved since its founding? \n \
                    \n \
                    Competitors: Who are the likely competitors of this company? What are their strengths and weaknesses? How does your company compare to its competitors in terms of offerings, market share, or other relevant factors?  \n \
                    \n \                
                    Mission & Vision: What is the company's mission statement or core purpose? What are the long-term goals and aspirations of the company? \n \
                    Values: What does the company value? What do they emphasize in their mission? What do they care about or prioritize? \n \
                    \n \
                  
                    NOTES ON FORMAT:
                    This should be at least 800 words. Be confident, do not say there is incomplete information, or there is not information. If you can't answer elements from the above, ignore it! Speak as if you are the authority of the subject. If you don't know the answer, don't talk about it. Do not say "I was unable to find information on XYZ". 
                    Ensure you keep the headers with the '--': 
                    -- Overview
                    (your overview)
                   
                    --Unique Value Proposition
                    (your response)
                    
                    --Competitors
                    (your response)
                    
                    -- Founding Story
                    (your response)
                    
                    --Mission & Vision
                    (your response)

                   --Values
                    (your response)
                    """})

    company_context = company_research['output']
    # Check if the output indicates insufficient information
    if "I couldn't find more information" in company_context:
        company_context = "Insufficient information. Please write the company description yourself."
    # Store the result in the task's state instead of returning it
    anvil.server.task_state['result'] = company_context

####### -------- PRODUCT --------###################################################

# @anvil.server.callable
# def launch_all_products_generator(company_profile, company_url):
#     print("Launch all products research function started")  
#     # Launch the background task
#     task = anvil.server.launch_background_task('all_products_generator', company_profile, company_url)
#     # Return the task ID
#     return task.get_id()

# @anvil.server.background_task
# def all_products_generator(company_profile, company_url):
#     print("Background task started for generating all the products:", company_profile, company_url)

#     llm_agents = ChatOpenAI(temperature=0.2, model_name='gpt-4', openai_api_key=openai_api_key)
#     agent_products_research = initialize_agent([tools], llm_agents, agent="zero-shot-react-description", handle_parsing_errors=True)
#     all_products_research = agent_products_research({"input": f""" You are ProductFinderAI, an advanced marketing consultant, your role is to guide a company determine their most popular and obvious products they should sell in order to boost their online presence, attract a larger customer base, and increase sales. You will employ the strategies of Russell Brunson, the founder of ClickFunnels, as detailed in his book "Dotcom Secrets". 
#     Your mission is to pinpoint five potential products or services that align best with the company's business. You should also rank these products or services based on their value to the company.
#     For each product or service, provide a title and a single sentence description that includes any other pertinent details (like pricing, access, special features, etc.) 
#     The output should be formatted as follows, and include "->" as a seperator which is very important!

#     'Title' -> Description of Product/Service 1. 
#     'Title' -> Description of Product/Service 2
#     'Title' -> Description of Product/Service 3
#     'Title' -> Description of Product/Service 4
#     'Title' -> Description of Product/Service 5
   
#     For instance: 
#     -- Freshsales Free CRM -> This plan is free for up to 3 users and includes a visual sales pipeline, automation via workflows and sales sequences, and built-in email, phone, and chat for contextual engagement. It provides everything you need to engage leads across phone, email & SMS.
#     -- Freshsales Growth ->Priced at $15/user/month when billed annually or $18/user/month when billed monthly, the Growth plan includes everything in the Free CRM plan, plus AI-powered contact scoring, up to 2,000 bot sessions per month, and sales sequences. It also includes 1 CPQ license. This plan is designed to help growing sales teams avoid repetitive work and spend more time selling.
#     -- Freshsales Pro -> Priced at $39/user/month when billed annually or $47/user/month when billed monthly, the Pro plan includes everything in the Growth plan, plus multiple sales pipelines, time-based workflows, AI-powered deal insights & next best action, up to 3,000 bot sessions per month, and sales teams & territory management. This plan is designed for managing multiple sales teams and growing revenue.
#     -- Freshsales Enterprise -> Priced at $69/user/month when billed annually or $83/user/month when billed monthly, the Enterprise plan includes everything in the Pro plan, plus custom modules, AI-based forecasting insights, audit logs, up to 5,000 bot sessions per month, and a dedicated account manager. This plan offers advanced customization, governance, and controls.
  
#     FORMAT: 
#     CONTEXTUAL INFORMATION:
#     COMPANY CONTEXT: {company_profile}
#     COMPANY WEBSITE: {company_url}
#     Chatbot:
#                     """})

#     all_products_grouped = all_products_research['output']
#     print("all_products_grouped:", all_products_grouped)
#     # # Check if the output indicates insufficient information
#     # if "I couldn't find more information" in all_products_research:
#     #     all_products_research = "Insufficient information. Please write the company description yourself."
#     # # Store the result in the task's state instead of returning it
   
#     all_product_lines = all_products_grouped.split("\n")

#     # Initialize an empty dictionary
#     all_products = {}

#     # Loop over each line
#     i = 1
#     for product_line in all_product_lines:
#         # Ignore empty lines
#         if not product_line.strip():
#             continue
    
#         # Check if the line starts with 'Ranking:'
#         if product_line.startswith('Ranking:'):
#             all_products['ranking'] = product_line.strip()
#         else:
#             # Split the line into title and description using '->' as the separator, if possible
#             line_parts = product_line.strip().split(' -> ')
#             if len(line_parts) >= 2:
#                 title, description = line_parts
#             else:
#                 # If the line doesn't contain the separator, consider the entire line as the title
#                 title = product_line.strip()
#                 description = ""  # Set description to an empty string or any default value
    
#             key = f"product_{i}"
#             value = f"{title} -> {description}"
    
#             # Add to dictionary
#             all_products[key] = value
#             i += 1

#       # # Return the resulting dictionary
#       # anvil.server.task_state['result'] = all_avatars

#     # Convert the dictionary to a JSON string
#     all_products_json = json.dumps(all_products)

#     # Return the resulting JSON string
#     anvil.server.task_state['result'] = all_products_json

#### LAUNCH THE PRODUCT DEEP DIVES

# PRODUCT 1
@anvil.server.callable
def launch_deepdive_product_1_generator(user_table,company_name,product_name,product_url,product_preview):
    # Launch the background task

  # START THE WEB SCRAPING
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
    page_content = requests.get(product_url, headers=headers).content

    soup = BeautifulSoup(page_content, "html.parser")
    # Extract all the text from the page
    bulky_text_content = soup.get_text()
   # Remove leading and trailing whitespaces, replace newlines and extra spaces
    product_webpage_scraped = bulky_text_content.strip().replace('\n', ' ').replace('\r', '').replace('  ', ' ')

    print("Scraped Information:",product_webpage_scraped)
  
    task = anvil.server.launch_background_task('deepdive_draft_product_1_generator',user_table,company_name,product_name,product_url,product_webpage_scraped)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def deepdive_product_1_generator(user_table,company_name,product_name,product_url,product_preview,product_webpage_scraped):
    print("Background task started for the Deep Dive of Researching the Product:", product_name)
    llm_agents = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo-16k', openai_api_key=openai_api_key)
    print("Background task started for generating the Product summary:", product_url)

    template_product_summary = """As a highly-skilled business research agent, your task is to conduct an exhaustive report and analysis of the company's product, {product_name} \
                  Leverage the product information that has been scraped from {company_name}'s' product website {product_url} in order to build your synopsis. However, note that there may be other products listed within the scraped information, so be diligent about your listed features. \
                  Lastly, be very specific! To help guide you, I'll provide a brief context about the product here: {product_preview}
                  
                  This is not an educational excercise. This work will be incorporated into our commercial operation shortly, so provide meaningful, actionable insights. Do not provide general terms or vague business ideas: be as particular about the issue as possible. Be confident. Provide numbers, statistics, prices, when possible!
                  \n \
                  Overview: Provide a comprehensive introduction to the product. What is its purpose, and what does the company aim to achieve with it? \n \
                  \n \
                  Description: Deeply describe the product. What does it look like, feel like, and what experience does it offer? \n \
                  \n \
                  Price: Detail the pricing structure. What is the cost, and are there any variations or tiers in pricing? \n \
                  \n \
                  Features: Elucidate the key features of the product. What distinguishes this product from others in the market? I would like around 15 differences between the product offers, if possible. \n \
                  \n \
                  Benefits: Explicate on how the product will benefit the customer. How can it change their life or improve their situation? \n \
                  \n \
                  Why people buy it: Analyze the consumer's pain points and desires before purchasing this product. Why might someone be drawn to this product, and what needs does it fulfill? \n \
                  \n \
                  Expected results: What are the anticipated outcomes or gains after using this product? How will the customer's situation improve or change? \n \
                  \n \
                  Guarantees: Discuss any guarantees the company offers with this product. Are there any assurances of product performance, or return policies in place? \n \
                  \n \
                  Bonuses: List any additional bonuses or incentives that come along with the product. What additional value does the company provide to sweeten the deal? \n \
                  \n \
                  Possible objections: Predict potential objections or concerns a customer may have about the product. How might the company address these? \n \
                  \n \
                  Ensure to provide an in-depth report with approximately 800-1000 words on the product, making it as detailed and specific as possible. Your aim is to capture the full essence of the product.
                  \n \
                  NOTES ON FORMAT:
                  Be confident. However, if there is incomplete information, please state "MORE INFORMATION NEEDED"! Speak as if you are the authority of the subject. 

                  ** END OF FORMAT
                  
                  FINALLY, HERE IS THE PRODUCT CONTEXT SCRAPED FROM THEIR PRODUCT WEBSITE: {product_webpage_scraped}
                  """
    
    prompt_product_summary = PromptTemplate(
        input_variables=["company_name", "product_name","product_url","product_preview","product_webpage_scraped"],
        template=template_product_summary
    )
  
    chain_product_summary = LLMChain(llm=llm_agents, prompt=prompt_product_summary)
    product_summary = chain_product_summary.run(company_name=company_name,product_name=product_name,product_url=product_url,product_preview=product_preview,product_webpage_scraped=product_webpage_scraped)  # Pass in the combined context
    
    print("PRODUCT SUMMARY:",product_summary)
  
   # Save it in the table:
    product_1_latest_row = user_table.search(variable='product_1_latest')[0]
    product_1_latest_row['variable_value'] = product_summary
    product_1_latest_row.update()
    print("Product Research Complete")


# PRODUCT 2
@anvil.server.callable
def launch_deepdive_product_2_generator(user_table,company_name,product_name,product_url,product_preview):
    # Launch the background task

  # START THE WEB SCRAPING
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
    page_content = requests.get(product_url, headers=headers).content

    soup = BeautifulSoup(page_content, "html.parser")
    # Extract all the text from the page
    bulky_text_content = soup.get_text()
   # Remove leading and trailing whitespaces, replace newlines and extra spaces
    product_webpage_scraped = bulky_text_content.strip().replace('\n', ' ').replace('\r', '').replace('  ', ' ')

    print("Scraped Information:",product_webpage_scraped)
  
    task = anvil.server.launch_background_task('deepdive_draft_product_2_generator',user_table,company_name,product_name,product_url,product_webpage_scraped)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def deepdive_product_2_generator(user_table,company_name,product_name,product_url,product_preview,product_webpage_scraped):
    print("Background task started for the Deep Dive of Researching the Product:", product_name)
    llm_agents = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo-16k', openai_api_key=openai_api_key)
    print("Background task started for generating the Product summary:", product_url)

    template_product_summary = """As a highly-skilled business research agent, your task is to conduct an exhaustive report and analysis of the company's product, {product_name} \
                  Leverage the product information that has been scraped from {company_name}'s' product website {product_url} in order to build your synopsis. However, note that there may be other products listed within the scraped information, so be diligent about your listed features. \
                  Lastly, be very specific! To help guide you, I'll provide a brief context about the product here: {product_preview}
                  
                  This is not an educational excercise. This work will be incorporated into our commercial operation shortly, so provide meaningful, actionable insights. Do not provide general terms or vague business ideas: be as particular about the issue as possible. Be confident. Provide numbers, statistics, prices, when possible!
                  \n \
                  Overview: Provide a comprehensive introduction to the product. What is its purpose, and what does the company aim to achieve with it? \n \
                  \n \
                  Description: Deeply describe the product. What does it look like, feel like, and what experience does it offer? \n \
                  \n \
                  Price: Detail the pricing structure. What is the cost, and are there any variations or tiers in pricing? \n \
                  \n \
                  Features: Elucidate the key features of the product. What distinguishes this product from others in the market? I would like around 15 differences between the product offers, if possible. \n \
                  \n \
                  Benefits: Explicate on how the product will benefit the customer. How can it change their life or improve their situation? \n \
                  \n \
                  Why people buy it: Analyze the consumer's pain points and desires before purchasing this product. Why might someone be drawn to this product, and what needs does it fulfill? \n \
                  \n \
                  Expected results: What are the anticipated outcomes or gains after using this product? How will the customer's situation improve or change? \n \
                  \n \
                  Guarantees: Discuss any guarantees the company offers with this product. Are there any assurances of product performance, or return policies in place? \n \
                  \n \
                  Bonuses: List any additional bonuses or incentives that come along with the product. What additional value does the company provide to sweeten the deal? \n \
                  \n \
                  Possible objections: Predict potential objections or concerns a customer may have about the product. How might the company address these? \n \
                  \n \
                  Ensure to provide an in-depth report with approximately 800-1000 words on the product, making it as detailed and specific as possible. Your aim is to capture the full essence of the product.
                  \n \
                  NOTES ON FORMAT:
                  Be confident. However, if there is incomplete information, please state "MORE INFORMATION NEEDED"! Speak as if you are the authority of the subject. 

                  ** END OF FORMAT
                  
                  FINALLY, HERE IS THE PRODUCT CONTEXT SCRAPED FROM THEIR PRODUCT WEBSITE: {product_webpage_scraped}
                  """
    
    prompt_product_summary = PromptTemplate(
        input_variables=["company_name", "product_name","product_url","product_preview","product_webpage_scraped"],
        template=template_product_summary
    )
  
    chain_product_summary = LLMChain(llm=llm_agents, prompt=prompt_product_summary)
    product_summary = chain_product_summary.run(company_name=company_name,product_name=product_name,product_url=product_url,product_preview=product_preview,product_webpage_scraped=product_webpage_scraped)  # Pass in the combined context
    
    print("PRODUCT SUMMARY:",product_summary)
  
   # Save it in the table:
    product_2_latest_row = user_table.search(variable='product_2_latest')[0]
    product_2_latest_row['variable_value'] = product_summary
    product_2_latest_row.update()
    print("Product Research Complete")

# PRODUCT 3
@anvil.server.callable
def launch_deepdive_product_3_generator(user_table,company_name,product_name,product_url,product_preview):
    # Launch the background task

  # START THE WEB SCRAPING
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
    page_content = requests.get(product_url, headers=headers).content

    soup = BeautifulSoup(page_content, "html.parser")
    # Extract all the text from the page
    bulky_text_content = soup.get_text()
   # Remove leading and trailing whitespaces, replace newlines and extra spaces
    product_webpage_scraped = bulky_text_content.strip().replace('\n', ' ').replace('\r', '').replace('  ', ' ')

    print("Scraped Information:",product_webpage_scraped)
  
    task = anvil.server.launch_background_task('deepdive_draft_product_3_generator',user_table,company_name,product_name,product_url,product_webpage_scraped)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def deepdive_product_3_generator(user_table,company_name,product_name,product_url,product_preview,product_webpage_scraped):
    print("Background task started for the Deep Dive of Researching the Product:", product_name)
    llm_agents = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo-16k', openai_api_key=openai_api_key)
    print("Background task started for generating the Product summary:", product_url)

    template_product_summary = """As a highly-skilled business research agent, your task is to conduct an exhaustive report and analysis of the company's product, {product_name} \
                  Leverage the product information that has been scraped from {company_name}'s' product website {product_url} in order to build your synopsis. However, note that there may be other products listed within the scraped information, so be diligent about your listed features. \
                  Lastly, be very specific! To help guide you, I'll provide a brief context about the product here: {product_preview}
                  
                  This is not an educational excercise. This work will be incorporated into our commercial operation shortly, so provide meaningful, actionable insights. Do not provide general terms or vague business ideas: be as particular about the issue as possible. Be confident. Provide numbers, statistics, prices, when possible!
                  \n \
                  Overview: Provide a comprehensive introduction to the product. What is its purpose, and what does the company aim to achieve with it? \n \
                  \n \
                  Description: Deeply describe the product. What does it look like, feel like, and what experience does it offer? \n \
                  \n \
                  Price: Detail the pricing structure. What is the cost, and are there any variations or tiers in pricing? \n \
                  \n \
                  Features: Elucidate the key features of the product. What distinguishes this product from others in the market? I would like around 15 differences between the product offers, if possible. \n \
                  \n \
                  Benefits: Explicate on how the product will benefit the customer. How can it change their life or improve their situation? \n \
                  \n \
                  Why people buy it: Analyze the consumer's pain points and desires before purchasing this product. Why might someone be drawn to this product, and what needs does it fulfill? \n \
                  \n \
                  Expected results: What are the anticipated outcomes or gains after using this product? How will the customer's situation improve or change? \n \
                  \n \
                  Guarantees: Discuss any guarantees the company offers with this product. Are there any assurances of product performance, or return policies in place? \n \
                  \n \
                  Bonuses: List any additional bonuses or incentives that come along with the product. What additional value does the company provide to sweeten the deal? \n \
                  \n \
                  Possible objections: Predict potential objections or concerns a customer may have about the product. How might the company address these? \n \
                  \n \
                  Ensure to provide an in-depth report with approximately 800-1000 words on the product, making it as detailed and specific as possible. Your aim is to capture the full essence of the product.
                  \n \
                  NOTES ON FORMAT:
                  Be confident. However, if there is incomplete information, please state "MORE INFORMATION NEEDED"! Speak as if you are the authority of the subject. 

                  ** END OF FORMAT
                  
                  FINALLY, HERE IS THE PRODUCT CONTEXT SCRAPED FROM THEIR PRODUCT WEBSITE: {product_webpage_scraped}
                  """
    
    prompt_product_summary = PromptTemplate(
        input_variables=["company_name", "product_name","product_url","product_preview","product_webpage_scraped"],
        template=template_product_summary
    )
  
    chain_product_summary = LLMChain(llm=llm_agents, prompt=prompt_product_summary)
    product_summary = chain_product_summary.run(company_name=company_name,product_name=product_name,product_url=product_url,product_preview=product_preview,product_webpage_scraped=product_webpage_scraped)  # Pass in the combined context
    
    print("PRODUCT SUMMARY:",product_summary)
  
   # Save it in the table:
    product_3_latest_row = user_table.search(variable='product_3_latest')[0]
    product_3_latest_row['variable_value'] = product_summary
    product_3_latest_row.update()
    print("Product Research Complete")

# PRODUCT 4
@anvil.server.callable
def launch_deepdive_product_4_generator(user_table,company_name,product_name,product_url,product_preview):
    # Launch the background task

  # START THE WEB SCRAPING
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
    page_content = requests.get(product_url, headers=headers).content

    soup = BeautifulSoup(page_content, "html.parser")
    # Extract all the text from the page
    bulky_text_content = soup.get_text()
   # Remove leading and trailing whitespaces, replace newlines and extra spaces
    product_webpage_scraped = bulky_text_content.strip().replace('\n', ' ').replace('\r', '').replace('  ', ' ')

    print("Scraped Information:",product_webpage_scraped)
  
    task = anvil.server.launch_background_task('deepdive_draft_product_4_generator',user_table,company_name,product_name,product_url,product_webpage_scraped)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def deepdive_product_4_generator(user_table,company_name,product_name,product_url,product_preview,product_webpage_scraped):
    print("Background task started for the Deep Dive of Researching the Product:", product_name)
    llm_agents = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo-16k', openai_api_key=openai_api_key)
    print("Background task started for generating the Product summary:", product_url)

    template_product_summary = """As a highly-skilled business research agent, your task is to conduct an exhaustive report and analysis of the company's product, {product_name} \
                  Leverage the product information that has been scraped from {company_name}'s' product website {product_url} in order to build your synopsis. However, note that there may be other products listed within the scraped information, so be diligent about your listed features. \
                  Lastly, be very specific! To help guide you, I'll provide a brief context about the product here: {product_preview}
                  
                  This is not an educational excercise. This work will be incorporated into our commercial operation shortly, so provide meaningful, actionable insights. Do not provide general terms or vague business ideas: be as particular about the issue as possible. Be confident. Provide numbers, statistics, prices, when possible!
                  \n \
                  Overview: Provide a comprehensive introduction to the product. What is its purpose, and what does the company aim to achieve with it? \n \
                  \n \
                  Description: Deeply describe the product. What does it look like, feel like, and what experience does it offer? \n \
                  \n \
                  Price: Detail the pricing structure. What is the cost, and are there any variations or tiers in pricing? \n \
                  \n \
                  Features: Elucidate the key features of the product. What distinguishes this product from others in the market? I would like around 15 differences between the product offers, if possible. \n \
                  \n \
                  Benefits: Explicate on how the product will benefit the customer. How can it change their life or improve their situation? \n \
                  \n \
                  Why people buy it: Analyze the consumer's pain points and desires before purchasing this product. Why might someone be drawn to this product, and what needs does it fulfill? \n \
                  \n \
                  Expected results: What are the anticipated outcomes or gains after using this product? How will the customer's situation improve or change? \n \
                  \n \
                  Guarantees: Discuss any guarantees the company offers with this product. Are there any assurances of product performance, or return policies in place? \n \
                  \n \
                  Bonuses: List any additional bonuses or incentives that come along with the product. What additional value does the company provide to sweeten the deal? \n \
                  \n \
                  Possible objections: Predict potential objections or concerns a customer may have about the product. How might the company address these? \n \
                  \n \
                  Ensure to provide an in-depth report with approximately 800-1000 words on the product, making it as detailed and specific as possible. Your aim is to capture the full essence of the product.
                  \n \
                  NOTES ON FORMAT:
                  Be confident. However, if there is incomplete information, please state "MORE INFORMATION NEEDED"! Speak as if you are the authority of the subject. 

                  ** END OF FORMAT
                  
                  FINALLY, HERE IS THE PRODUCT CONTEXT SCRAPED FROM THEIR PRODUCT WEBSITE: {product_webpage_scraped}
                  """
    
    prompt_product_summary = PromptTemplate(
        input_variables=["company_name", "product_name","product_url","product_preview","product_webpage_scraped"],
        template=template_product_summary
    )
  
    chain_product_summary = LLMChain(llm=llm_agents, prompt=prompt_product_summary)
    product_summary = chain_product_summary.run(company_name=company_name,product_name=product_name,product_url=product_url,product_preview=product_preview,product_webpage_scraped=product_webpage_scraped)  # Pass in the combined context
    
    print("PRODUCT SUMMARY:",product_summary)
  
   # Save it in the table:
    product_4_latest_row = user_table.search(variable='product_4_latest')[0]
    product_4_latest_row['variable_value'] = product_summary
    product_4_latest_row.update()
    print("Product Research Complete")

# PRODUCT 5
@anvil.server.callable
def launch_deepdive_product_5_generator(user_table,company_name,product_name,product_url,product_preview):
    # Launch the background task

  # START THE WEB SCRAPING
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
    page_content = requests.get(product_url, headers=headers).content

    soup = BeautifulSoup(page_content, "html.parser")
    # Extract all the text from the page
    bulky_text_content = soup.get_text()
   # Remove leading and trailing whitespaces, replace newlines and extra spaces
    product_webpage_scraped = bulky_text_content.strip().replace('\n', ' ').replace('\r', '').replace('  ', ' ')

    print("Scraped Information:",product_webpage_scraped)
  
    task = anvil.server.launch_background_task('deepdive_draft_product_5_generator',user_table,company_name,product_name,product_url,product_webpage_scraped)
    # Return the task ID
    return task.get_id()
  
@anvil.server.background_task
def deepdive_product_5_generator(user_table,company_name,product_name,product_url,product_preview,product_webpage_scraped):
    print("Background task started for the Deep Dive of Researching the Product:", product_name)
    llm_agents = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo-16k', openai_api_key=openai_api_key)
    print("Background task started for generating the Product summary:", product_url)

    template_product_summary = """As a highly-skilled business research agent, your task is to conduct an exhaustive report and analysis of the company's product, {product_name} \
                  Leverage the product information that has been scraped from {company_name}'s' product website {product_url} in order to build your synopsis. However, note that there may be other products listed within the scraped information, so be diligent about your listed features. \
                  Lastly, be very specific! To help guide you, I'll provide a brief context about the product here: {product_preview}
                  
                  This is not an educational excercise. This work will be incorporated into our commercial operation shortly, so provide meaningful, actionable insights. Do not provide general terms or vague business ideas: be as particular about the issue as possible. Be confident. Provide numbers, statistics, prices, when possible!
                  \n \
                  Overview: Provide a comprehensive introduction to the product. What is its purpose, and what does the company aim to achieve with it? \n \
                  \n \
                  Description: Deeply describe the product. What does it look like, feel like, and what experience does it offer? \n \
                  \n \
                  Price: Detail the pricing structure. What is the cost, and are there any variations or tiers in pricing? \n \
                  \n \
                  Features: Elucidate the key features of the product. What distinguishes this product from others in the market? I would like around 15 differences between the product offers, if possible. \n \
                  \n \
                  Benefits: Explicate on how the product will benefit the customer. How can it change their life or improve their situation? \n \
                  \n \
                  Why people buy it: Analyze the consumer's pain points and desires before purchasing this product. Why might someone be drawn to this product, and what needs does it fulfill? \n \
                  \n \
                  Expected results: What are the anticipated outcomes or gains after using this product? How will the customer's situation improve or change? \n \
                  \n \
                  Guarantees: Discuss any guarantees the company offers with this product. Are there any assurances of product performance, or return policies in place? \n \
                  \n \
                  Bonuses: List any additional bonuses or incentives that come along with the product. What additional value does the company provide to sweeten the deal? \n \
                  \n \
                  Possible objections: Predict potential objections or concerns a customer may have about the product. How might the company address these? \n \
                  \n \
                  Ensure to provide an in-depth report with approximately 800-1000 words on the product, making it as detailed and specific as possible. Your aim is to capture the full essence of the product.
                  \n \
                  NOTES ON FORMAT:
                  Be confident. However, if there is incomplete information, please state "MORE INFORMATION NEEDED"! Speak as if you are the authority of the subject. 

                  ** END OF FORMAT
                  
                  FINALLY, HERE IS THE PRODUCT CONTEXT SCRAPED FROM THEIR PRODUCT WEBSITE: {product_webpage_scraped}
                  """
    
    prompt_product_summary = PromptTemplate(
        input_variables=["company_name", "product_name","product_url","product_preview","product_webpage_scraped"],
        template=template_product_summary
    )
  
    chain_product_summary = LLMChain(llm=llm_agents, prompt=prompt_product_summary)
    product_summary = chain_product_summary.run(company_name=company_name,product_name=product_name,product_url=product_url,product_preview=product_preview,product_webpage_scraped=product_webpage_scraped)  # Pass in the combined context
    
    print("PRODUCT SUMMARY:",product_summary)
  
   # Save it in the table:
    product_5_latest_row = user_table.search(variable='product_5_latest')[0]
    product_5_latest_row['variable_value'] = product_summary
    product_5_latest_row.update()
    print("Product Research Complete")
  
####### -------- BRAND TONE --------###################################################

@anvil.server.callable
def launch_brand_tone_research(brand_tone_url):
    # Launch the background task
    task = anvil.server.launch_background_task('brand_tone_research',brand_tone_url)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def brand_tone_research(brand_tone_url):
    print("Background task started for extracting brand tone:", brand_tone_url)
 
    llm_agents = ChatOpenAI(temperature=0.2, model_name='gpt-4', openai_api_key=openai_api_key)
    agent_tone_extraction = initialize_agent([tools], llm_agents, agent="zero-shot-react-description", handle_parsing_errors=True)
 
    tone_research = agent_tone_extraction({
      "input":f"""You are CopywriterAI, the best copywriter on the planet. We are looking to generate a description 
      of the tone that best describes the copywriting style and tone of an existing web page. Go and research this URL 
      {brand_tone_url}, and provide your analysis.

      For example, for PROFESSIONAL / ENTERPRISE, it would be described as:
        - 'Formal and Polished': (sophisticated language and complex sentence structures).
        - 'Objective and Analytical': (incorporates data and facts, prioritizing logical arguments).
        - 'Business-like': Efficient, frequently employs industry-specific jargon and business terminology).
        - 'Trustworthy and Reliable': Underscoring credibility, reliability, and accuracy.
        - 'Instructional and Informative': (providing clear, direct instructions or information).
        - 'Respectful and Considerate': (acknowledging the audience's needs and viewpoints while avoiding excessive casualness).
        - 'Controlled and Consistent': (providing coherence, ensuring careful, consistent writing).

        For Russell Brunson, sales page style, it would be described as:
        - 'Conversational': (friendly, casual, and approachable).
        - 'Storytelling': (using compelling stories to illustrate his points).
        - 'Educational': (being informative, teaching something new).
        - 'Persuasive': (being compelling and enticing, using ideas of scarcity (limited time offers), social proof (testimonials), and authority (expertise and success).
        - 'Inspiring': (motivating and inspiring, encouraging the reader to take action).
        - 'Clear and Direct': (providing clarity and simplicity, avoiding jargon).

        However, it is up to you to go and review the website, think about the tone of the existing copy, and return 5-6 descriptors, in the similar format as above. They don't have to be listed above- they can be new!
        
        OUTPUT TEMPLATE: AN EXAMPLE OUTPUT SHOULD BE AS BELOW:
        'The business tone for {brand_tone_url} can be described as': 
        - 'Conversational': (friendly, casual, and approachable).
        - 'Storytelling': (using compelling stories to illustrate his points).
        - 'Educational': (being informative, teaching something new).
        - 'Persuasive': (being compelling and enticing, using ideas of scarcity (limited time offers), social proof (testimonials), and authority (expertise and success).
        - 'Inspiring': (motivating and inspiring, encouraging the reader to take action).
        - 'Clear and Direct': (providing clarity and simplicity, avoiding jargon).Conversational, Storytelling, Educational, Persuasive, Inspiring, Clear and Direct'
        
        FINAL RULES: Don't mention the business name, or source. Just say "the business" and refer to it as 'company tone' or 'the business tone'
        """})

    extracted_tone = tone_research['output']
    anvil.server.task_state['result'] = extracted_tone

@anvil.server.callable
def save_brand_tone_component_click(self, **event_args):
    # Get the current user
    current_user = anvil.users.get_user()
    # Get the email of the current user
    owner = current_user['email']

    # Get the row for the current user from the variable_table
    row = app_tables.variable_table.get(owner=owner)  # Replace user_email with owner
    if row:
        text = self.brand_tonetextbox.text
        row['brand_tone'] = text
        row.update()
    else:
        # Handle case where the row does not exist for the current user
        print("No row found for the current user")

####### -------- AVATARS --------###################################################

# GENERATE ALL 5 AVATAR OUTLINES ---#####

@anvil.server.callable
def launch_all_avatars_generator(owner_company_profile):
    print("Launch all Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('all_avatars_generator', owner_company_profile)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def all_avatars_generator(owner_company_profile):
    print("Background task started for generating all the avatars:", owner_company_profile)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_all_avatars = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.
    Your task is to provide the company with a five different archetypal customer avatars as they best relates to their business. However, it is your role to identify which avatars are most valuable for the company, and rank them in order.
    For each of the five avatars, provide a single title of the avatar, followed by a single sentence description of the avatar including name, age, location, and any other important info that we'll break down later. 
    The format for the output will be as follows:

    Title - Description of Avatar 1
    Title - Description of Avatar 2
    Title - Description of Avatar 3
    Title - Description of Avatar 4
    Title - Description of Avatar 5
   
    For example: 
    - The Analyzer - Amy is a 34-year-old entrepreneur based in New York City who runs a successful e-commerce business and is always looking for ways to optimize his marketing strategies and increase revenue. He is tech-savvy and data-driven, and values tools that provide actionable insights and help him make informed decisions.
    - The Novice - John is a 28-year-old small business owner based in a rural area who is looking to expand her business online. She has limited experience with digital marketing and is looking for a user-friendly tool that can guide her through the process of optimizing her marketing strategies and increasing her online presence.
    ----
    FORMAT: 
    CONTEXTUAL INFORMATION:

    COMPANY CONTEXT: {owner_company_profile}

    Chatbot:"""

    prompt_all_avatars = PromptTemplate(
        input_variables=["owner_company_profile"],
        template=template_all_avatars 
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_all_avatars)
    all_avatars_grouped = chain_avatar.run(owner_company_profile=owner_company_profile)
    # anvil.server.task_state['result'] = all_avatars_grouped
    
    lines = all_avatars_grouped.split("\n")

    # Initialize an empty dictionary
    all_avatars = {}

    # Loop over each line
    i = 1
    for line in lines:
        # Ignore empty lines
        if not line.strip():
            continue

        # Check if the line starts with 'Ranking:'
        if line.startswith('Ranking:'):
            all_avatars['ranking'] = line.strip()
        else:
            # Split the line into title and description
            title, description = line.strip().split(' - ')
            key = f"avatar_{i}"
            value = f"{title} - {description}"

            # Add to dictionary
            all_avatars[key] = value
            i += 1

      # # Return the resulting dictionary
      # anvil.server.task_state['result'] = all_avatars

    # Convert the dictionary to a JSON string
    all_avatars_json = json.dumps(all_avatars)

    # Return the resulting JSON string
    anvil.server.task_state['result'] = all_avatars_json

#------GENERATE SINGLE AVATAR, SPECIFIC TO A PRODUCT: AVATAR X_PRODUCT_Y.....---------#################

# AVATAR 1, PRODUCT 1 -----------------------#################
@anvil.server.callable
def launch_deepdive_avatar_1_product_1_generator(product_1_name,product_1_profile,avatar_1_product_1_name_preview,avatar_1_product_1_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task

    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='avatar_1_product_1_latest')

    task = anvil.server.launch_background_task('deepdive_avatar_1_product_1_generator', product_1_name,product_1_profile,avatar_1_product_1_name_preview,avatar_1_product_1_preview,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_1_product_1_generator(product_1_name,product_1_profile,avatar_1_product_1_name_preview,avatar_1_product_1_preview,row):
    print("Background task started for generating the avatar:", avatar_1_product_1_name_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. We'd like to create the ideal customer avatar for a product. \
    Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    We're looking to create the ideal customer avatar for the following product: {product_1_name}
    The product is described as: {product_1_profile}
    The avatar's name is: {avatar_1_product_1_name_preview}
    A brief description of the avatar to expand on is: {avatar_1_product_1_preview}
    
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details above, as it best relates to their business, broken down as follows:

    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["product_1_name","product_1_profile","avatar_1_product_1_name_preview","avatar_1_product_1_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(product_1_name=product_1_name,product_1_profile=product_1_profile,avatar_1_product_1_name_preview=avatar_1_product_1_name_preview,avatar_1_product_1_preview=avatar_1_product_1_preview)  # Pass in the combined context

   # Save the generated avatar in the 'avatar latest' column of the variable_table
    row['variable_value'] = avatar
    row.update()
  
    anvil.server.task_state['result']  = avatar

# AVATAR 2, PRODUCT 1 -----##
@anvil.server.callable
def launch_deepdive_avatar_2_product_1_generator(product_1_name,product_1_profile,avatar_2_product_1_name_preview,avatar_2_product_1_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    
    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='avatar_2_product_1_latest')
    
    task = anvil.server.launch_background_task('deepdive_avatar_2_product_1_generator', product_1_name,product_1_profile,avatar_2_product_1_name_preview,avatar_2_product_1_preview,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_2_product_1_generator(product_1_name,product_1_profile,avatar_2_product_1_name_preview,avatar_2_product_1_preview,row):
    print("Background task started for generating the avatar:", avatar_2_product_1_name_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. We'd like to create the ideal customer avatar for a product. \
    Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    We're looking to create the ideal customer avatar for the following product: {product_1_name}
    The product is described as: {product_1_profile}
    The avatar's name is: {avatar_2_product_1_name_preview}
    A brief description of the avatar to expand on is: {avatar_2_product_1_preview}
    
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details above, as it best relates to their business, broken down as follows:

    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["product_1_name","product_1_profile","avatar_2_product_1_name_preview","avatar_2_product_1_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(product_1_name=product_1_name,product_1_profile=product_1_profile,avatar_2_product_1_name_preview=avatar_2_product_1_name_preview,avatar_2_product_1_preview=avatar_2_product_1_preview)  # Pass in the combined context
     # Save the generated avatar in the 'avatar latest' column of the variable_table
    row['variable_value'] = avatar
    row.update()
  
    anvil.server.task_state['result']  = avatar

# AVATAR 3, PRODUCT 1 -----##
@anvil.server.callable
def launch_deepdive_avatar_3_product_1_generator(product_1_name,product_1_profile,avatar_3_product_1_name_preview,avatar_3_product_1_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
  
    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='avatar_3_product_1_latest')
    
    task = anvil.server.launch_background_task('deepdive_avatar_3_product_1_generator', product_1_name,product_1_profile,avatar_3_product_1_name_preview,avatar_3_product_1_preview,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_3_product_1_generator(product_1_name,product_1_profile,avatar_3_product_1_name_preview,avatar_3_product_1_preview,row):
    print("Background task started for generating the avatar:", avatar_3_product_1_name_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. We'd like to create the ideal customer avatar for a product. \
    Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    We're looking to create the ideal customer avatar for the following product: {product_1_name}
    The product is described as: {product_1_profile}
    The avatar's name is: {avatar_3_product_1_name_preview}
    A brief description of the avatar to expand on is: {avatar_3_product_1_preview}
    
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details above, as it best relates to their business, broken down as follows:

    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["product_1_name","product_1_profile","avatar_3_product_1_name_preview","avatar_3_product_1_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(product_1_name=product_1_name,product_1_profile=product_1_profile,avatar_3_product_1_name_preview=avatar_3_product_1_name_preview,avatar_3_product_1_preview=avatar_3_product_1_preview)  # Pass in the combined context
     
    # Save the generated avatar in the 'avatar latest' column of the variable_table
    row['variable_value'] = avatar
    row.update()
  
    anvil.server.task_state['result']  = avatar

#------PRODUCT 2------------------------#################

# AVATAR 1, PRODUCT 2 -----##
@anvil.server.callable
def launch_deepdive_avatar_1_product_2_generator(product_2_name,product_2_profile,avatar_1_product_2_name_preview,avatar_1_product_2_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    
    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='avatar_1_product_2_latest')
   
    task = anvil.server.launch_background_task('deepdive_avatar_1_product_2_generator', product_2_name,product_2_profile,avatar_1_product_2_name_preview,avatar_1_product_2_preview, row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_1_product_2_generator(product_2_name,product_2_profile,avatar_1_product_2_name_preview,avatar_1_product_2_preview,row):
    print("Background task started for generating the avatar:", avatar_1_product_2_name_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. We'd like to create the ideal customer avatar for a product. \
    Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    We're looking to create the ideal customer avatar for the following product: {product_2_name}
    The product is described as: {product_2_profile}
    The avatar's name is: {avatar_1_product_2_name_preview}
    A brief description of the avatar to expand on is: {avatar_1_product_2_preview}
    
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details above, as it best relates to their business, broken down as follows:

    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["product_2_name","product_2_profile","avatar_1_product_2_name_preview","avatar_1_product_2_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(product_2_name=product_2_name,product_2_profile=product_2_profile,avatar_1_product_2_name_preview=avatar_1_product_2_name_preview,avatar_1_product_2_preview=avatar_1_product_2_preview)  # Pass in the combined context

     # Save the generated avatar in the 'avatar latest' column of the variable_table
    row['variable_value'] = avatar
    row.update()
 
    anvil.server.task_state['result']  = avatar

# AVATAR 2, PRODUCT 2 -----##
@anvil.server.callable
def launch_deepdive_avatar_2_product_2_generator(product_2_name,product_2_profile,avatar_2_product_2_name_preview,avatar_2_product_2_preview):
    print("Launch Deep Dive Avatar function started")  
     
    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='avatar_2_product_2_latest')
  
    # Launch the background task
    task = anvil.server.launch_background_task('deepdive_avatar_2_product_2_generator', product_2_name,product_2_profile,avatar_2_product_2_name_preview,avatar_2_product_2_preview,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_2_product_2_generator(product_2_name,product_2_profile,avatar_2_product_2_name_preview,avatar_2_product_2_preview,row):
    print("Background task started for generating the avatar:", avatar_2_product_2_name_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. We'd like to create the ideal customer avatar for a product. \
    Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    We're looking to create the ideal customer avatar for the following product: {product_2_name}
    The product is described as: {product_2_profile}
    The avatar's name is: {avatar_2_product_2_name_preview}
    A brief description of the avatar to expand on is: {avatar_2_product_2_preview}
    
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details above, as it best relates to their business, broken down as follows:

    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["product_2_name","product_2_profile","avatar_2_product_2_name_preview","avatar_2_product_2_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(product_2_name=product_2_name,product_2_profile=product_2_profile,avatar_2_product_2_name_preview=avatar_2_product_2_name_preview,avatar_2_product_2_preview=avatar_2_product_2_preview)  # Pass in the combined context

     # Save the generated avatar in the 'avatar latest' column of the variable_table
    row['variable_value'] = avatar
    row.update()
    
    anvil.server.task_state['result']  = avatar

# AVATAR 3, PRODUCT 2 ---------------##
@anvil.server.callable
def launch_deepdive_avatar_3_product_2_generator(product_2_name,product_2_profile,avatar_3_product_2_name_preview,avatar_3_product_2_preview):
    print("Launch Deep Dive Avatar function started")  

    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='avatar_3_product_2_latest')
  
    # Launch the background task
    task = anvil.server.launch_background_task('deepdive_avatar_3_product_2_generator', product_2_name,product_2_profile,avatar_3_product_2_name_preview,avatar_3_product_2_preview,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_3_product_2_generator(product_2_name,product_2_profile,avatar_3_product_2_name_preview,avatar_3_product_2_preview,row):
    print("Background task started for generating the avatar:", avatar_3_product_2_name_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. We'd like to create the ideal customer avatar for a product. \
    Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    We're looking to create the ideal customer avatar for the following product: {product_2_name}
    The product is described as: {product_2_profile}
    The avatar's name is: {avatar_3_product_2_name_preview}
    A brief description of the avatar to expand on is: {avatar_3_product_2_preview}
    
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details above, as it best relates to their business, broken down as follows:

    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["product_2_name","product_2_profile","avatar_3_product_2_name_preview","avatar_3_product_2_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(product_2_name=product_2_name,product_2_profile=product_2_profile,avatar_3_product_2_name_preview=avatar_3_product_2_name_preview,avatar_3_product_2_preview=avatar_3_product_2_preview)  # Pass in the combined context
     
    # Save the generated avatar in the 'avatar latest' column of the variable_table
    row['variable_value'] = avatar
    row.update()
  
    anvil.server.task_state['result']  = avatar

#------PRODUCT 3------------------------#################

# AVATAR 1, PRODUCT 3 -----##
@anvil.server.callable
def launch_deepdive_avatar_1_product_3_generator(product_3_name,product_3_profile,avatar_1_product_3_name_preview,avatar_1_product_3_preview):
    print("Launch Deep Dive Avatar function started") 
    
    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='avatar_1_product_3_latest')

    # Launch the background task
    task = anvil.server.launch_background_task('deepdive_avatar_1_product_3_generator', product_3_name,product_3_profile,avatar_1_product_3_name_preview,avatar_1_product_3_preview,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_1_product_3_generator(product_3_name,product_3_profile,avatar_1_product_3_name_preview,avatar_1_product_3_preview,row):
    print("Background task started for generating the avatar:", avatar_1_product_3_name_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. We'd like to create the ideal customer avatar for a product. \
    Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    We're looking to create the ideal customer avatar for the following product: {product_3_name}
    The product is described as: {product_3_profile}
    The avatar's name is: {avatar_1_product_3_name_preview}
    A brief description of the avatar to expand on is: {avatar_1_product_3_preview}
    
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details above, as it best relates to their business, broken down as follows:

    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["product_3_name","product_3_profile","avatar_1_product_3_name_preview","avatar_1_product_3_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(product_3_name=product_3_name,product_3_profile=product_3_profile,avatar_1_product_3_name_preview=avatar_1_product_3_name_preview,avatar_1_product_3_preview=avatar_1_product_3_preview)  # Pass in the combined context
    
   # Save the generated avatar in the 'avatar latest' column of the variable_table
    row['variable_value'] = avatar
    row.update()
  
    anvil.server.task_state['result']  = avatar

# AVATAR 2, PRODUCT 3 -----##
@anvil.server.callable
def launch_deepdive_avatar_2_product_3_generator(product_3_name,product_3_profile,avatar_2_product_3_name_preview,avatar_2_product_3_preview):
    print("Launch Deep Dive Avatar function started") 

    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='avatar_2_product_3_latest')

    # Launch the background task
    task = anvil.server.launch_background_task('deepdive_avatar_2_product_2_generator', product_2_name,product_3_profile,avatar_2_product_2_name_preview,avatar_2_product_2_preview,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_2_product_3_generator(product_3_name,product_3_profile,avatar_2_product_3_name_preview,avatar_2_product_3_preview,row):
    print("Background task started for generating the avatar:", avatar_2_product_2_name_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. We'd like to create the ideal customer avatar for a product. \
    Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    We're looking to create the ideal customer avatar for the following product: {product_2_name}
    The product is described as: {product_2_profile}
    The avatar's name is: {avatar_2_product_2_name_preview}
    A brief description of the avatar to expand on is: {avatar_2_product_2_preview}
    
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details above, as it best relates to their business, broken down as follows:

    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["product_2_name","product_2_profile","avatar_2_product_2_name_preview","avatar_2_product_2_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(product_2_name=product_2_name,product_2_profile=product_2_profile,avatar_2_product_2_name_preview=avatar_2_product_2_name_preview,avatar_2_product_2_preview=avatar_2_product_2_preview)  # Pass in the combined context
    
  # Save the generated avatar in the 'avatar latest' column of the variable_table
    row['variable_value'] = avatar
    row.update()
  
    anvil.server.task_state['result']  = avatar

# AVATAR 3, PRODUCT 3 ---------------##
@anvil.server.callable
def launch_deepdive_avatar_3_product_3_generator(product_3_name,product_3_profile,avatar_3_product_3_name_preview,avatar_3_product_3_preview):
    print("Launch Deep Dive Avatar function started")

    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='avatar_3_product_3_latest')

    # Launch the background task
    task = anvil.server.launch_background_task('deepdive_avatar_3_product_2_generator', product_2_name,product_2_profile,avatar_3_product_2_name_preview,avatar_3_product_2_preview,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_3_product_3_generator(product_3_name,product_2_profile,avatar_3_product_3_name_preview,avatar_3_product_3_preview,row):
    print("Background task started for generating the avatar:", avatar_3_product_3_name_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. We'd like to create the ideal customer avatar for a product. \
    Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    We're looking to create the ideal customer avatar for the following product: {product_3_name}
    The product is described as: {product_3_profile}
    The avatar's name is: {avatar_3_product_3_name_preview}
    A brief description of the avatar to expand on is: {avatar_3_product_3_preview}
    
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details above, as it best relates to their business, broken down as follows:

    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["product_3_name","product_3_profile","avatar_3_product_3_name_preview","avatar_3_product_3_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(product_3_name=product_3_name,product_3_profile=product_3_profile,avatar_3_product_3_name_preview=avatar_3_product_3_name_preview,avatar_3_product_3_preview=avatar_3_product_3_preview)  # Pass in the combined context
   
    # Save the generated avatar in the 'avatar latest' column of the variable_table
    row['variable_value'] = avatar
    row.update()
  
    anvil.server.task_state['result']  = avatar

#------PRODUCT 4------------------------#################

# AVATAR 1, PRODUCT 4 -----##
@anvil.server.callable
def launch_deepdive_avatar_1_product_4_generator(product_4_name,product_4_profile,avatar_1_product_4_name_preview,avatar_1_product_4_preview):
    print("Launch Deep Dive Avatar function started")  

    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='avatar_1_product_4_latest')

    # Launch the background task
    task = anvil.server.launch_background_task('deepdive_avatar_1_product_3_generator', product_4_name,product_4_profile,avatar_1_product_4_name_preview,avatar_1_product_4_preview,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_1_product_4_generator(product_4_name,product_4_profile,avatar_1_product_4_name_preview,avatar_1_product_4_preview,row):
    print("Background task started for generating the avatar:", avatar_1_product_3_name_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. We'd like to create the ideal customer avatar for a product. \
    Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    We're looking to create the ideal customer avatar for the following product: {product_3_name}
    The product is described as: {product_3_profile}
    The avatar's name is: {avatar_1_product_3_name_preview}
    A brief description of the avatar to expand on is: {avatar_1_product_3_preview}
    
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details above, as it best relates to their business, broken down as follows:

    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["product_3_name","product_3_profile","avatar_1_product_3_name_preview","avatar_1_product_3_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(product_3_name=product_3_name,product_3_profile=product_3_profile,avatar_1_product_3_name_preview=avatar_1_product_3_name_preview,avatar_1_product_3_preview=avatar_1_product_3_preview)  # Pass in the combined context
     
    # Save the generated avatar in the 'avatar latest' column of the variable_table
    row['variable_value'] = avatar
    row.update()
    
    anvil.server.task_state['result']  = avatar

# AVATAR 2, PRODUCT 4 -----##
@anvil.server.callable
def launch_deepdive_avatar_2_product_4_generator(product_4_name,product_4_profile,avatar_2_product_4_name_preview,avatar_2_product_4_preview):
    print("Launch Deep Dive Avatar function started")  

    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='avatar_2_product_4_latest')

    # Launch the background task
    task = anvil.server.launch_background_task('deepdive_avatar_2_product_4_generator', product_4_name,product_4_profile,avatar_2_product_4_name_preview,avatar_2_product_4_preview,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_2_product_4_generator(product_4_name,product_4_profile,avatar_2_product_4_name_preview,avatar_2_product_4_preview,row):
    print("Background task started for generating the avatar:", avatar_2_product_4_name_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. We'd like to create the ideal customer avatar for a product. \
    Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    We're looking to create the ideal customer avatar for the following product: {product_2_name}
    The product is described as: {product_2_profile}
    The avatar's name is: {avatar_2_product_2_name_preview}
    A brief description of the avatar to expand on is: {avatar_2_product_2_preview}
    
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details above, as it best relates to their business, broken down as follows:

    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["product_4_name","product_4_profile","avatar_2_product_4_name_preview","avatar_2_product_4_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(product_4_name=product_4_name,product_4_profile=product_4_profile,avatar_2_product_4_name_preview=avatar_2_product_4_name_preview,avatar_2_product_4_preview=avatar_2_product_4_preview)  # Pass in the combined context
    
    # Save the generated avatar in the 'avatar latest' column of the variable_table
    row['variable_value'] = avatar
    row.update()
  
    anvil.server.task_state['result']  = avatar

# AVATAR 3, PRODUCT 4 ---------------##
@anvil.server.callable
def launch_deepdive_avatar_3_product_4_generator(product_4_name,product_4_profile,avatar_3_product_4_name_preview,avatar_3_product_4_preview):
    print("Launch Deep Dive Avatar function started")  

    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='avatar_3_product_4_latest')

    # Launch the background task
    task = anvil.server.launch_background_task('deepdive_avatar_4_product_2_generator', product_4_name,product_4_profile,avatar_3_product_4_name_preview,avatar_3_product_4_preview,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_3_product_4_generator(product_4_name,product_2_profile,avatar_3_product_4_name_preview,avatar_3_product_4_preview,row):
    print("Background task started for generating the avatar:", avatar_3_product_4_name_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. We'd like to create the ideal customer avatar for a product. \
    Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    We're looking to create the ideal customer avatar for the following product: {product_4_name}
    The product is described as: {product_4_profile}
    The avatar's name is: {avatar_3_product_4_name_preview}
    A brief description of the avatar to expand on is: {avatar_3_product_4_preview}
    
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details above, as it best relates to their business, broken down as follows:

    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["product_4_name","product_4_profile","avatar_3_product_4_name_preview","avatar_3_product_4_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(product_4_name=product_4_name,product_4_profile=product_4_profile,avatar_3_product_4_name_preview=avatar_3_product_4_name_preview,avatar_3_product_4_preview=avatar_3_product_4_preview)  # Pass in the combined context
    
     # Save the generated avatar in the 'avatar latest' column of the variable_table
    row['variable_value'] = avatar
    row.update()

    anvil.server.task_state['result']  = avatar

#------PRODUCT 5------------------------#################

# AVATAR 1, PRODUCT 5 -----------------------#################
@anvil.server.callable
def launch_deepdive_avatar_1_product_5_generator(product_5_name,product_5_profile,avatar_1_product_5_name_preview,avatar_1_product_5_preview):
    print("Launch Deep Dive Avatar function started")  

    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='avatar_1_product_5_latest')

    # Launch the background task
    task = anvil.server.launch_background_task('deepdive_avatar_1_product_5_generator', product_5_name,product_5_profile,avatar_1_product_5_name_preview,avatar_1_product_5_preview,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_1_product_5_generator(product_5_name,product_5_profile,avatar_1_product_5_name_preview,avatar_1_product_5_preview,row):
    print("Background task started for generating the avatar:", avatar_1_product_5_name_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. We'd like to create the ideal customer avatar for a product. \
    Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    We're looking to create the ideal customer avatar for the following product: {product_5_name}
    The product is described as: {product_5_profile}
    The avatar's name is: {avatar_1_product_5_name_preview}
    A brief description of the avatar to expand on is: {avatar_1_product_5_preview}
    
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details above, as it best relates to their business, broken down as follows:

    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["product_5_name","product_5_profile","avatar_1_product_5_name_preview","avatar_1_product_5_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(product_5_name=product_5_name,product_5_profile=product_5_profile,avatar_1_product_5_name_preview=avatar_1_product_5_name_preview,avatar_1_product_5_preview=avatar_1_product_5_preview)  # Pass in the combined context

    # Save the generated avatar in the 'avatar latest' column of the variable_table
    row['variable_value'] = avatar
    row.update()

    anvil.server.task_state['result']  = avatar

# AVATAR 2, PRODUCT 5-----##
@anvil.server.callable
def launch_deepdive_avatar_2_product_5_generator(product_5_name,product_5_profile,avatar_2_product_5_name_preview,avatar_2_product_5_preview):
    print("Launch Deep Dive Avatar function started")  
    
    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='avatar_2_product_5_latest')

    # Launch the background task
    task = anvil.server.launch_background_task('deepdive_avatar_2_product_5_generator', product_5_name,product_5_profile,avatar_2_product_5_name_preview,avatar_2_product_5_preview,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_2_product_5_generator(product_5_name,product_5_profile,avatar_2_product_5_name_preview,avatar_2_product_5_preview,row):
    print("Background task started for generating the avatar:", avatar_2_product_5_name_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. We'd like to create the ideal customer avatar for a product. \
    Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    We're looking to create the ideal customer avatar for the following product: {product_5_name}
    The product is described as: {product_5_profile}
    The avatar's name is: {avatar_2_product_5_name_preview}
    A brief description of the avatar to expand on is: {avatar_2_product_5_preview}
    
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details above, as it best relates to their business, broken down as follows:

    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["product_5_name","product_5_profile","avatar_2_product_5_name_preview","avatar_2_product_5_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(product_5_name=product_5_name,product_5_profile=product_5_profile,avatar_2_product_5_name_preview=avatar_1_product_5_name_preview,avatar_2_product_5_preview=avatar_2_product_5_preview)  # Pass in the combined context
    
   # Save the generated avatar in the 'avatar latest' column of the variable_table
    row['variable_value'] = avatar
    row.update()
  
    anvil.server.task_state['result']  = avatar

# AVATAR 3, PRODUCT 5 -----##
@anvil.server.callable
def launch_deepdive_avatar_3_product_5_generator(product_5_name,product_5_profile,avatar_3_product_5_name_preview,avatar_3_product_5_preview):
    print("Launch Deep Dive Avatar function started")  

    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='avatar_3_product_5_latest')

    # Launch the background task
    task = anvil.server.launch_background_task('deepdive_avatar_3_product_5_generator', product_5_name,product_5_profile,avatar_3_product_5_name_preview,avatar_3_product_5_preview,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_3_product_5_generator(product_5_name,product_5_profile,avatar_3_product_5_name_preview,avatar_3_product_5_preview,row):
    print("Background task started for generating the avatar:", avatar_3_product_5_name_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. We'd like to create the ideal customer avatar for a product. \
    Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    We're looking to create the ideal customer avatar for the following product: {product_5_name}
    The product is described as: {product_5_profile}
    The avatar's name is: {avatar_3_product_5_name_preview}
    A brief description of the avatar to expand on is: {avatar_3_product_5_preview}
    
    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details above, as it best relates to their business, broken down as follows:

    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["product_5_name","product_5_profile","avatar_3_product_5_name_preview","avatar_3_product_5_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(product_5_name=product_5_name,product_5_profile=product_5_profile,avatar_3_product_5_name_preview=avatar_3_product_5_name_preview,avatar_3_product_5_preview=avatar_3_product_5_preview)  # Pass in the combined context
    
    # Save the generated avatar in the 'avatar latest' column of the variable_table
    row['variable_value'] = avatar
    row.update()
  
    anvil.server.task_state['result']  = avatar
  

#------CREATE GENERIC  AVATARS FOR THE COMPANY---------#################

@anvil.server.callable
def launch_deepdive_avatar_1_generator(owner_company_profile,avatar_1_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('deepdive_avatar_1_generator', owner_company_profile,avatar_1_preview)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_1_generator(owner_company_profile,avatar_1_preview):
    print("Background task started for generating the avatar:", owner_company_profile,avatar_1_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    A brief description of the avatar to expand on is: {avatar_1_preview}

    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:

    Here's the Avatar Preview to base the Full Avatar for: {avatar_1_preview}
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    COMPANY CONTEXT: {owner_company_profile}
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["owner_company_profile", "avatar_1_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(owner_company_profile=owner_company_profile, avatar_1_preview=avatar_1_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = avatar

# AVATAR 2
@anvil.server.callable
def launch_deepdive_avatar_2_generator(owner_company_profile,avatar_2_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('deepdive_avatar_2_generator', owner_company_profile,avatar_2_preview)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_2_generator(owner_company_profile,avatar_2_preview):
    print("Background task started for generating the avatar:", owner_company_profile,avatar_2_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    A brief description of the avatar to expand on is: {avatar_2_preview}

    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:

    Here's the Avatar Preview to base the Full Avatar for: {avatar_2_preview}
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    COMPANY CONTEXT: {owner_company_profile}
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["owner_company_profile", "avatar_2_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(owner_company_profile=owner_company_profile, avatar_2_preview=avatar_2_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = avatar

# AVATAR 3
@anvil.server.callable
def launch_deepdive_avatar_3_generator(owner_company_profile,avatar_3_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('deepdive_avatar_3_generator', owner_company_profile,avatar_3_preview)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_3_generator(owner_company_profile,avatar_3_preview):
    print("Background task started for generating the avatar:", owner_company_profile,avatar_3_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    A brief description of the avatar to expand on is: {avatar_3_preview}

    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:

    Here's the Avatar Preview to base the Full Avatar for: {avatar_3_preview}
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    COMPANY CONTEXT: {owner_company_profile}
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["owner_company_profile", "avatar_3_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(owner_company_profile=owner_company_profile, avatar_3_preview=avatar_3_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = avatar

# AVATAR 4
@anvil.server.callable
def launch_deepdive_avatar_4_generator(owner_company_profile,avatar_4_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('deepdive_avatar_4_generator', owner_company_profile,avatar_4_preview)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_4_generator(owner_company_profile,avatar_4_preview):
    print("Background task started for generating the avatar:", owner_company_profile,avatar_4_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    A brief description of the avatar to expand on is: {avatar_4_preview}

    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:

    Here's the Avatar Preview to base the Full Avatar for: {avatar_4_preview}
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    COMPANY CONTEXT: {owner_company_profile}
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["owner_company_profile", "avatar_4_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(owner_company_profile=owner_company_profile, avatar_4_preview=avatar_4_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = avatar

# AVATAR 5
@anvil.server.callable
def launch_deepdive_avatar_5_generator(owner_company_profile,avatar_5_preview):
    print("Launch Deep Dive Avatar function started")  
    # Launch the background task
    task = anvil.server.launch_background_task('deepdive_avatar_5_generator', owner_company_profile,avatar_5_preview)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def deepdive_avatar_5_generator(owner_company_profile,avatar_5_preview):
    print("Background task started for generating the avatar:", owner_company_profile,avatar_5_preview)
 
    llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
    template_avatar = """You are AvatarAI, the most advanced marketing consultant in the world. You are advising a company who is looking to grow their presence online, attract customers and sell more units. To help them do this, you reference and abide by the concepts of Russell Brunson, the founder of ClickFunnels, in his book "Dotcom Secrets", and approach our exercise the same way Russell Brunson would. Below, we'll provide the format of how we'd like the question answered, as well as the contextual information.

    A brief description of the avatar to expand on is: {avatar_5_preview}

    Your task is to provide the company with a detailed customer avatar based on the short avatar preview details below, as it best relates to their business, broken down as follows:

    Here's the Avatar Preview to base the Full Avatar for: {avatar_5_preview}
    ----
    FORMAT: 
    - Overview
    Provide a comprehensive summary of the typical customer for the company, outlining their key characteristics.

    - Demographic
    Provide specific demographic data on the target customer, including age, gender, location, income level, education level, and occupation.

    - Psychographic
    Provide detailed information about the psychological attributes of the avatar, such as their interests, attitudes, values, and lifestyle preferences. Use exampples, not hypotheticals.

    - Goals & Aspirations
    Provide a brief synopsis of the avatars personal and professional goals, dreams, and aspirations.

    - Pain Points
    Identify the specific problems, challenges, and frustrations the avatar is facing.

    - Personal Experience
    Provide insights into the personal experiences of the avatar that shapes their preferences, behaviors, and decisions, including their past interactions with similar products or services. Provide real world examples.

    RULES: 
    - Do not say "the target customer", instead, provide a fictional name, age, location.  
    - Don't be general...we are looking for very specific avatars! If you don't know the answer, make an educated creative guess. Be as detailed and specific as possible!
    - Do not explain theory...paint us a picture with an example. This isn't an education lesson, it's a practical exercise.
    -----
    COMPANY CONTEXT: {owner_company_profile}
    
    Chatbot:"""

    prompt_avatar = PromptTemplate(
        input_variables=["owner_company_profile", "avatar_5_preview"],
        template=template_avatar
    )

    chain_avatar = LLMChain(llm=llm_agents, prompt=prompt_avatar)
    avatar = chain_avatar.run(owner_company_profile=owner_company_profile, avatar_5_preview=avatar_5_preview)  # Pass in the combined context
    anvil.server.task_state['result']  = avatar


### SAVING
@anvil.server.callable
def save_avatar(owner, avatar_number, avatar):
    # Get the row for the current user from the variable_table
    row = app_tables.variable_table.get(owner=owner)
    if row:
        text = avatar
        row[avatar_number] = text  # Use the variable avatar_number directly
        row.update()
    else:
        # Handle case where the row does not exist for the current user
        print("No row found for the current user")

####### -------- LOCK IN VARIABLES --------###################################################

@anvil.server.callable
def get_chosen_variable_value(user_table, selected_variable_title):
    chosen_variable_value = None
    if selected_variable_title:
        matching_rows = user_table.search(variable_title=selected_variable_title)
        if matching_rows:
            chosen_variable_value = matching_rows[0]['variable_value']
    return chosen_variable_value

@anvil.server.callable
def get_chosen_variable_avatar(user_table, selected_variable_value):
    chosen_variable_value = None
    if selected_variable_value:
        matching_rows = user_table.search(variable_value=selected_variable_value)
        if matching_rows:
            chosen_variable_value = matching_rows[0]['variable_value']
    return chosen_variable_value

@anvil.server.callable
def save_funnel_settings_component(user_table_name, selected_company_profile_value, selected_product_name_value):
    user_table = getattr(tables, user_table_name)
    chosen_company_profile = get_chosen_variable_value(user_table, selected_company_profile_value)
    chosen_product_name = get_chosen_variable_value(user_table, selected_product_name_value)
    # Perform further operations or return the values as needed
    return chosen_company_profile, chosen_product_name
  

####### -------- HEADLINES --------###################################################

# This is the headline generator that will return a string of 10 of the best headlines
@anvil.server.callable
def launch_generate_main_headlines(chosen_product_name, chosen_company_profile, chosen_product_research, chosen_tone):
    print("Launch Generate Main Headlines Function") 

    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='main_headlines')
  
  
    # Launch the background task
    task = anvil.server.launch_background_task('generate_main_headlines',chosen_product_name, chosen_company_profile, chosen_product_research, chosen_tone,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def generate_main_headlines(chosen_product_name, chosen_company_profile, chosen_product_research, chosen_tone, row):
  # example_headlines_row = app_tables.example_scripts.get(script='example_headlines')
  # example_headlines = example_headlines_row['script_contents']
  # HERE IS ARE SOME EXAMPLE HEADLINES: {example_headlines}

  llm_headline = ChatOpenAI(temperature=0.8, model_name='gpt-4', openai_api_key=openai_api_key)
  
  headline_template = """ You are RussellAI, a highly-evolved version of Russell Brunson, the author and business coach behind "Dotcom Secrets". You are the best copywriter on the planet. You are about to launch a brand new marketing funnel selling {chosen_product_name}, and you need to generate the best attention grabbing headlines that stir curiosity and compel potential users to learn more about {chosen_product_name}.  
    
    First, I will provide you some context about the company, and then I will provide you some context about the product, and then I will give you many examples of headlines from a parallel industry for you to apply to our product. Next, you will generate 10 of the most incredible, mind-blowing headlines that will stop people in their tracks and want to learn more about {chosen_product_name}, but you must adapt the headlines to be in the tone I provide.

    When generating these headlines just remember that people didn’t come looking for our product… instead we are interrupting them in their daily journey. The only way to get them to stop scrolling online is to grab their attention with an irresistible headline!

    HERE IS SOME CONTEXT ABOUT THE COMPANY: {chosen_company_profile}

    HERE IS SOME CONTEXT ABOUT THE PRODUCT: {chosen_product_research}
    
             
    THE TONE IS: {chosen_tone}

    The output should be an unumbered list of 10 headlines, as per the tone I provide. Update the example headlines I gave according to the tone.
    Finally, lead with the best one at the top! (no introduction or outro needed). 
    No "" quotation marks.
    No itemized numbers. Just text.
    Output should be in a similar format:
    Finally! Unmask the Power of Data: Transform Your Marketing with Funnelytics Performance!
    NOT "Finally! Unmask the Power of Data: Transform Your Marketing with Funnelytics Performance!"
    """

  headline_prompt = PromptTemplate(
      input_variables=["chosen_product_name", "chosen_company_profile", "chosen_product_research", "chosen_tone"], 
      template=headline_template
  )
  
  chain_main_headlines = LLMChain(llm=llm_headline, prompt=headline_prompt)
  headline_generator = chain_main_headlines.run(chosen_product_name=chosen_product_name, chosen_company_profile=chosen_company_profile, chosen_product_research=chosen_product_research, chosen_tone=chosen_tone)
  print("Here are the headlines", headline_generator) 
  headlines = headline_generator.split("\n")

  # Initialize an empty list
  all_main_headlines = []

 # Loop over each line
  for headline in headlines:
      # Ignore empty lines
      if not headline.strip():
          continue

      # Append the headline to the list
      all_main_headlines.append(headline.strip())

  # Convert the list to a JSON string
  all_main_headlines_json = json.dumps(all_main_headlines)

  # Save the generated headlines in the 'main_headlines' column of the variable_table
  row['variable_value'] = all_main_headlines_json
  row.update()

  # Return the resulting JSON string
  anvil.server.task_state['result'] = all_main_headlines_json

  
####### --------SUB HEADLINES --------###################################################

# This is the subheadline generator that will return a string of 10 of the best subheadlines
@anvil.server.callable
def launch_generate_subheadlines(chosen_product_name, chosen_company_profile, chosen_product_research, chosen_tone):
    print("Launch Generate SubHeadlines Function") 
    
    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='subheadlines')
    task = anvil.server.launch_background_task('generate_subheadlines', chosen_product_name, chosen_company_profile, chosen_product_research, chosen_tone,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def generate_subheadlines(chosen_product_name, chosen_company_profile, chosen_product_research, chosen_tone,row):
  # example_headlines_row = app_tables.example_scripts.get(script='example_headlines')
  # example_headlines = example_headlines_row['script_contents'] 
  # HERE IS ARE SOME EXAMPLE HEADLINES, HOWEVER, YOU MUST UPDATE THEM TO MATCH THE TONE: {example_headlines}

  llm_subheadline = ChatOpenAI(temperature=0.8, model_name='gpt-4', openai_api_key=openai_api_key)
  
  subheadline_template = """  You are RussellAI, a highly-evolved version of Russell Brunson, the author and business coach behind "Dotcom Secrets". You are the best copywriter on the planet. You are about to launch a brand new marketing funnel selling {chosen_product_name}, and you've already received some attention grabbing headlines that stir curiosity and compel potential users to learn more about {chosen_product_name}. However, you now need to generate some sublines to support your main headlines.

    First, I will provide you the existing headlines, then I will provide you some context about the company, and then I will provide you some context about the product. Therafter, I will give you many examples of main headlines (not sublines) from a parallel industry (becoming an author) for you to understand the tone. Finally, for each of the headlines provided in the list, you will generate a single incredible, mind-blowing subheadline corresponding to the main headline that will stop people in their tracks and want to learn more about {chosen_product_name}.

    Here are some principles for writing compelling sublines:
    - Match the Headline: The subline should logically continue the thought or the promise made in the headline. It must be consistent in tone and message.
    - Highlight Key Benefits: Sublines often provide a space to explain the primary advantages or unique features of the product or service. Think of what makes your offer irresistible or different and emphasize it.
    - Target Audience: Make it clear who your product or service is for. If the headline hasn't done this, the subline should.
    - Provide Context or Explain: If the headline is designed to create intrigue or curiosity, the subline can provide enough information to encourage the reader to continue to engage.
    - Call to Action: While not always the case, sometimes a subline can provide a mild call to action or create a sense of urgency.
    - Keep it Brief: While a subline can be longer than a headline, it should still be succinct and easy to read at a glance.

    When generating these headlines just remember that people didn’t come looking for our product… instead we are interrupting them in their daily journey. The only way to get them to stop scrolling online is to grab their attention with an irresistible headline!

    HERE IS SOME CONTEXT ABOUT THE COMPANY: {chosen_company_profile}

    HERE IS SOME CONTEXT ABOUT THE PRODUCT: {chosen_product_research}
    
   
    THE TONE IS: {chosen_tone}
    
    The output should be a list of 10 SUBHEADLINES, in the tone above, that relate to the final existing main headline.
    
    No "" quotation marks.
    No itemized numbers. 
    Do not list them like '1.' '2.'... Just text.
    For example: 'BREAKING NEWS! Eliminate the Guesswork!'instead of '"1. Breaking News! Eliminate the Guesswork!"''
    (no introduction or outro needed, just an itemized list of 10 subheadlines)
    """

  subheadline_prompt = PromptTemplate(
      input_variables=["chosen_product_name", "chosen_company_profile", "chosen_product_research", "chosen_tone"], 
      template=subheadline_template
  )

  chain_subheadlines = LLMChain(llm=llm_subheadline, prompt=subheadline_prompt)
  subheadline_generator = chain_subheadlines.run(chosen_product_name=chosen_product_name, chosen_company_profile=chosen_company_profile, chosen_product_research=chosen_product_research, chosen_tone=chosen_tone)
  print("Here are the subheadlines", subheadline_generator) 
  subheadlines = subheadline_generator.split("\n")

  # Initialize an empty list
  all_subheadlines = []

 # Loop over each line
  for subheadline in subheadlines:
      # Ignore empty lines
      if not subheadline.strip():
          continue

      # Append the headline to the list
      all_subheadlines.append(subheadline.strip())

  # Convert the list to a JSON string
  all_subheadlines_json = json.dumps(all_subheadlines)

  # Save the generated subheadlines in the 'subheadlines' column of the variable_table
  row['variable_value'] = all_subheadlines_json
  row.update()

  # Return the resulting JSON string
  anvil.server.task_state['result'] = all_subheadlines_json


####### --------VIDEO SALES SCRIPT --------###################################################

@anvil.server.callable
def launch_generate_vsl_script(chosen_product_name, chosen_company_profile, chosen_product_research, chosen_avatar, chosen_tone, example_script):
    print("Launch Generate Video Sales Letter Script Function")
    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='vsl_script')
  
    # Launch the background task
    task = anvil.server.launch_background_task('generate_vsl_script', chosen_product_name, chosen_company_profile, chosen_product_research, chosen_avatar, chosen_tone, example_script,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def generate_vsl_script(chosen_product_name, chosen_company_profile, chosen_product_research, chosen_avatar, chosen_tone,example_script,row):
    # Return the task ID):
    print("Background task started for generating the Video Sales Letter script")

    llm_vsl_script = ChatOpenAI(temperature=0.8, model_name='gpt-4', openai_api_key=openai_api_key)

    vsl_script_template = """You are RussellAI, a highly-evolved version of Russell Brunson, the author and business coach behind "Dotcom Secrets". You are the best scriptwriter on the planet. You are about to launch a brand new video sales letter marketing funnel selling {chosen_product_name}, and you're ready to start write the video sales letter script! This has a very specific format, and requires a lot of context, provided below:
     
    First, I will provide you with some tips about writing, then I will give you the existing headlines, some context about the company, the ideal customer we're trying to serve, followed by information about the product. Therafter, I will provide you some existing sales scripts (from a parallel industry) that will inform you of style and length. Lastly, I'll request this in an certain order, and provide you with a template to follow.
    TIPS: This script helps build credibility quickly by answering the problems our avatar faces, provides credibility, explains the product, then gives reasons to act now. It's important to remember that people didn’t come looking for our product… instead we are interrupting them in their daily journey. The only way to get them to stop scrolling online is to grab their attention with an irresistible scripts!

    HERE IS SOME CONTEXT ABOUT THE COMPANY: {chosen_company_profile}
    HERE IS SOME CONTEXT ABOUT THE PRODUCT: {chosen_product_research}

    HERE IS THE EXISTING CUSTOMER: {chosen_avatar}
    
    HERE ARE SOME EXAMPLES OF EXISTING SCRIPTS FROM PARALELL INDUSTRIES. YOU MUST UPDATE IT ACCORDING TO OUR PRODUCT AND COMPANY CONTEXT: {example_script} 

    TONE: {chosen_tone}

    HERE IS THE TEMPLATE TO FOLLOW WHEN CREATING THE SCRIPT:
    Explain The Problem — What problem is our avatar and target market facing? How can we empathize with their challenges? (should be between 90-100 words)
    Agitate The Problem — What are some examples of that problem? Make that problem visceral for them. Explain why it’s a bigger problem than they think it is and how it’s really going to harm them over the long-run. (should be between 90-100 words)
    Introduce The Solution — What is your solution to their problem? It's our product, of course! (should be between 90-100 words)
    Build Credibility — Why should they trust our founder to be the provider of this solution? Use their name. What makes you so great? Telling a story about your own journey can help build credibility. (should be between 90-100 words)
    Show Proof — How do they know that it’ll actually work? Make up a fictional case-study using ficticious details. This is important to discuss and show proof. (should be between 90-100 words)
    Explain Exactly What They Get — Explain exactly what the prospect is going to get if they sign up! (should be between 90-100 words)
    Give Reason To Act Now — Why should they buy right now? Use urgency or scarcity to put the prospect’s foot on the gas.(should be between 90-100 words)
    Close — Close the sale with a final call-to-action. 

    Lastly, NEVER mention you are RussellAI. Use the founders name of the company, or make up a name.
    The output should be a script, written in the first person from the perspective of the founder that is trying to sell the audience on why their product is the best choice and will make their life easier. The script should not include any subheadings!"""

    
    vsl_script_prompt = PromptTemplate(
        input_variables=["chosen_product_name", "chosen_company_profile", "chosen_product_research", "chosen_avatar", "chosen_tone","example_script"],
        template=vsl_script_template
    )

    chain_vsl_script = LLMChain(llm=llm_vsl_script, prompt=vsl_script_prompt)
    vsl_script = chain_vsl_script.run(chosen_product_name=chosen_product_name, chosen_company_profile=chosen_company_profile,chosen_product_research=chosen_product_research,chosen_avatar=chosen_avatar, chosen_tone=chosen_tone,example_script=example_script)

      # Save the generated subheadlines in the 'subheadlines' column of the variable_table
    
    row['variable_value'] = vsl_script
    row.update()
    anvil.server.task_state['result'] = vsl_script

####### --------VIDEO SALES SCRIPT WITH FEEDBACK

@anvil.server.callable
def launch_generate_vsl_script_with_feedback(chosen_product_name, chosen_product_research,vsl_script_feedback):
    print("Launch Generate Video Sales Letter Script Function")
    current_user = anvil.users.get_user()
    user_table_name = current_user['user_id']
    # Get the table for the current user
    user_table = getattr(app_tables, user_table_name)
    row = user_table.get(variable='vsl_script')
  
    # Launch the background task
    task = anvil.server.launch_background_task('generate_vsl_script_with_feedback', chosen_product_name, chosen_product_research,vsl_script_feedback,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def generate_vsl_script_with_feedback(chosen_product_name, chosen_product_research,vsl_script_feedback,row):
    # Return the task ID):
    print("Background task started for generating the Video Sales Letter script")

    llm_vsl_script = ChatOpenAI(temperature=0.8, model_name='gpt-4', openai_api_key=openai_api_key)

    vsl_script_template = """You are RussellAI, a highly-evolved version of Russell Brunson, the author and business coach behind "Dotcom Secrets". You are the best scriptwriter on the planet, and you've just written an amazing and effect script that will surely convert. This script will help you sell {chosen_product_name} to more people than ever!
     
    However, your ad partner, a truly gifted marketer with great understanding of the customers needs, has given some minor feedback they would like you to incorporate into your existing script (shared below). The format and structure is very important to maintain. You should maintain the existing tone, cadence, names and details, but modify the script according to the notes received.
    
    Remember, this is a sales script. Do your best to address the notes, and improve the script as much as you can.

    HERE IS SOME CONTEXT ABOUT THE PRODUCT: {chosen_product_research}

    HERE ARE THE NOTES YOU'VE RECEIVED FROM YOUR AD PARTNER. {vsl_script_feedback} 

    HERE IS THE TEMPLATE TO ADHERE TO WHEN EDITING OR MODIIFYING THE EXISTING SCRIPT.
    Explain The Problem — What problem is our avatar and target market facing? How can we empathize with their challenges? (should be between 90-100 words)
    Agitate The Problem — What are some examples of that problem? Make that problem visceral for them. Explain why it’s a bigger problem than they think it is and how it’s really going to harm them over the long-run. (should be between 90-100 words)
    Introduce The Solution — What is your solution to their problem? It's our product, of course! (should be between 90-100 words)
    Build Credibility — Why should they trust our founder to be the provider of this solution? Use their name. What makes you so great? Telling a story about your own journey can help build credibility. (should be between 90-100 words)
    Show Proof — How do they know that it’ll actually work? Make up a fictional case-study using ficticious details. This is important to discuss and show proof. (should be between 90-100 words)
    Explain Exactly What They Get — Explain exactly what the prospect is going to get if they sign up! (should be between 90-100 words)
    Give Reason To Act Now — Why should they buy right now? Use urgency or scarcity to put the prospect’s foot on the gas.(should be between 90-100 words)
    Close — Close the sale with a final call-to-action. 

    Lastly, NEVER mention you are RussellAI. Use the founders name of the company, or make up a name.
    The output should be a script, written in the first person from the perspective of the founder that is trying to sell the audience on why their product is the best choice and will make their life easier. The script should not include any subheadings!"""

    vsl_script_prompt = PromptTemplate(
        input_variables=["chosen_product_name", "chosen_product_research","vsl_script_feedback"],
        template=vsl_script_template
    )

    chain_vsl_script = LLMChain(llm=llm_vsl_script, prompt=vsl_script_prompt)
    vsl_script = chain_vsl_script.run(chosen_product_name=chosen_product_name, chosen_product_research=chosen_product_research,vsl_script_feedback=vsl_script_feedback)

      # Save the generated subheadlines in the 'subheadlines' column of the variable_table
    
    row['variable_value'] = vsl_script
    row.update()
    anvil.server.task_state['result'] = vsl_script
####### --------VIDEO SALES SCRIPT 4 THEMES --------###################################################

@anvil.server.callable 
def launch_generate_vsl_themes(chosen_final_headline, chosen_final_subheadline, chosen_product_name, chosen_product_research, chosen_tone,vsl_script,row):
    print("Launch Generate VSL Themes Function") 
    task = anvil.server.launch_background_task('generate_vsl_themes',chosen_final_headline,chosen_final_subheadline,chosen_product_name, chosen_product_research, chosen_tone,vsl_script,row)
    # Return the task ID
    return task.get_id()

@anvil.server.background_task
def generate_vsl_themes(chosen_final_headline,chosen_final_subheadline, chosen_product_name, chosen_product_research, chosen_tone,vsl_script,row):
  llm_vsl_themes = ChatOpenAI(temperature=0.8, model_name='gpt-4', openai_api_key=openai_api_key)
  four_vsl_themes_template = """ You are RussellAI, a highly-evolved version of Russell Brunson, the author and business coach behind "Dotcom Secrets". You are the best scriptwriter on the planet. You are about to launch a brand new video sales letter marketing funnel selling {chosen_product_name}, and you've already generated the sales video letter script, but you now need to extract the four themes from the script and promote them as the industry's dirty secret that will make them millions! These will be captions to screenshots from the video.

    These extractions help build credibility quickly by addressing the pain points of our customer, provides credibility, explains the product, then gives reasons to act now. It's important to remember that people didn’t come looking for our product… instead we are interrupting them in their daily journey. The only way to get them to stop scrolling online is to grab their attention with an irresistible scripts!
    First, I will provide you with video's main headline then some context about the product. Therafter, I will provide you with the final script that I need you to summarize and extract themes and reveal the big secrets of our product. Lastly, I'll request this in an certain order, and provide you with a template to follow.
    
    INGEST THE BELOW INFORMATION WITH SQUARE BRACKETS AS CONTEXT:
    [
    EXISTING HEADLINES: {chosen_final_headline}, plus {chosen_final_subheadline}

    CONTEXT ABOUT THE PRODUCT: {chosen_product_research}

    HFINAL SCRIPT OF THE VIDEO I NEED YOU TO EXTRACT THE BIG SECRETS FROM: {vsl_script}
    
    TONE: {chosen_tone}

    DO NOT INCLUDE ANY SUMMARIZATION OF THE ABOVE POINTS IN THE OUTPUT. I AM ONLY INTERESTED IN THE BELOW OUTPUT:
    
    ----- FINAL OUTPUT IS BELOW-----HERE IS THE TEMPLATE TO FOLLOW WHEN CREATING THE 4 EXCERPTS

    "SECRET #1:" 5-7 words of the theme or secret reveal, but in the form of a cheeky and confident headline. Then, provide an exciting sentence about how to be successful in that area, then trail off with an ellipses like this ....
    "SECRET #2:" 5-7 words of the theme or secret reveal,but in the form of a cheeky and confident headline. Then, provide an exciting sentence about how to be successful in that area, then trail off with an ellipses like this ....
    "SECRET #3:" 5-7 words of the theme or secret reveal, but in the form of a cheeky and confident headline. Then, provide an exciting sentence about how to be successful in that area, then trail off with an ellipses like this ....
    "SECRET #4:" a mini-headline that is 5-7 words of what, but  can be next in how they apply these themes. Then, provide a sentence about the magic results they could see..
    
    For example, the a potential output could look like below.

    SECRET #1: 'Attribution Agitation' at its Worst: Fed up with not knowing where your sales are coming from? Discover how Funnelytics Performance can clear up the confusion for good...

    SECRET #2: 'Ad-Cost Anguish' Annihilated: Struggling with soaring ad costs and sub-par results? Learn how to optimize your campaigns and slash ad spend with our innovative platform...

    SECRET #3: 'Funnel Failure' Flipped Upside Down: Tired of ineffective marketing funnels that just don't deliver? Watch as Funnelytics Performance revamps your funnel strategies and turns them into massive growth engines...

    SECRET #4: 'Scaling Struggles' Solved: Wondering how to grow your business without breaking the bank? Witness the magic as Funnelytics Performance helps you unlock unprecedented growth and skyrocket your success...'
    ]

    NOTE: The Final Output will be just Secret 1 through 4. NOT EVEN QUOTE MARKS "". Nothing else!
    THAT'S IT...NOW GO AND CREATE THE 4 EXCERPTS!
    --

    SECRET #1:...
    SECRET #2:...
    SECRET #3:...
    SECRET #4:..
    """

  vsl_themes_prompt = PromptTemplate(
      input_variables=["chosen_final_headline","chosen_final_subheadline","chosen_product_name", "chosen_product_research", "chosen_tone","vsl_script"], 
      template=four_vsl_themes_template
  )

  chain_vsl_themes = LLMChain(llm=llm_vsl_themes, prompt=vsl_themes_prompt)
  vsl_themes_generator = chain_vsl_themes.run(chosen_final_headline=chosen_final_headline, chosen_final_subheadline=chosen_final_subheadline,chosen_product_name=chosen_product_name, chosen_product_research=chosen_product_research, chosen_tone=chosen_tone,vsl_script=vsl_script)
  print("Here are the 4 excerpts", vsl_themes_generator) 
  vsl_themes = vsl_themes_generator.split("\n")

  # Initialize an empty list
  all_vsl_themes = []

 # Loop over each line
  for vsl_theme in vsl_themes:
      # Ignore empty lines
      if not vsl_theme.strip():
          continue

      # Append the headline to the list
      all_vsl_themes.append(vsl_theme.strip())

  # Convert the list to a JSON string
  all_vsl_themes_json = json.dumps(all_vsl_themes)

  row['variable_value'] = all_vsl_themes_json
  row.update()

  # Return the resulting JSON string
  anvil.server.task_state['result'] = all_vsl_themes_json
  
####### -------- TASK UPDATES --------###################################################
# # BACKGROUND UPDATES
# @anvil.server.callable
# def get_task_status(task_id):
#     # Get the task object by its ID
#     task = anvil.server.get_background_task(task_id)
#     # Return the termination status of the task
#     return task.get_termination_status()

@anvil.server.callable
def get_task_status(task_id):
    # Get the task object by its ID
    task = anvil.server.get_background_task(task_id)
    # Return the termination status of the task
    return task.get_termination_status()
  


# @anvil.server.callable
# def get_task_status(task_id):
#     try:
#         # Get the task object by its ID
#         task = anvil.server.get_background_task(task_id)
#         # Return the termination status of the task
#         return task.get_termination_status()
#     except anvil.server.BackgroundTaskNotFound:
#         return "not_found"
      

@anvil.server.callable
def get_task_result(task_id):
    # Get the task object by its ID
    task = anvil.server.get_background_task(task_id)
    # Get the task's state
    task_state = task.get_state()
    # Return the result of the task if it exists, otherwise return None
    return task_state.get('result')


# OLD CODE

# PRODUCT 3 DEEPDIVE
# @anvil.server.callable
# def launch_deepdive_product_3_generator(company_name,company_profile,company_url,product_3_name,product_3_preview):
#     # Launch the background task
#     task = anvil.server.launch_background_task('deepdive_product_3_generator',company_name, company_profile,company_url,product_3_name,product_3_preview)
#     # Return the task ID
#     return task.get_id()
  
# @anvil.server.background_task
# def deepdive_product_3_generator(company_name,company_profile,company_url,product_3_name,product_3_preview):
  
#     print("Background task started for the Deep Dive of Researching the Product:", product_3_name)

#     llm_agents = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=openai_api_key)
#     agent_product_research = initialize_agent([tools], llm_agents, agent="zero-shot-react-description", handle_parsing_errors=True)
  
#     product_research_context = agent_product_research({"input": f"""As a highly-skilled business research agent, your task is to conduct an exhaustive report and analysis of the company's product, {product_3_name} \
#                   Leverage all necessary resources such as {company_name}'s' website, {company_url}, web pages, and any other relevant sources \
#                   to gather the following details about company's product, {product_3_name}. Lastly, be very specific! This is not an educational excercise. This work will be incorporated into our commercial operation shortly, so provide meaningful, actionable insights. Do not provide general terms or vague business ideas: be as particular about the issue as possible. Be confident. Provide numbers, statistics, prices, when possible!
#                   \n \
#                   Overview: Provide a comprehensive introduction to the product. What is its purpose, and what does the company aim to achieve with it? \n \
#                   \n \
#                   Description: Deeply describe the product. What does it look like, feel like, and what experience does it offer? \n \
#                   \n \
#                   Price: Detail the pricing structure. What is the cost, and are there any variations or tiers in pricing? \n \
#                   \n \
#                   Features: Elucidate the key features of the product. What distinguishes this product from others in the market? I would like around 15 differences between the product offers, if possible. \n \
#                   \n \
#                   Benefits: Explicate on how the product will benefit the customer. How can it change their life or improve their situation? \n \
#                   \n \
#                   Why people buy it: Analyze the consumer's pain points and desires before purchasing this product. Why might someone be drawn to this product, and what needs does it fulfill? \n \
#                   \n \
#                   Expected results: What are the anticipated outcomes or gains after using this product? How will the customer's situation improve or change? \n \
#                   \n \
#                   Guarantees: Discuss any guarantees the company offers with this product. Are there any assurances of product performance, or return policies in place? \n \
#                   \n \
#                   Bonuses: List any additional bonuses or incentives that come along with the product. What additional value does the company provide to sweeten the deal? \n \
#                   \n \
#                   Possible objections: Predict potential objections or concerns a customer may have about the product. How might the company address these? \n \
#                   \n \
#                   Ensure to provide an in-depth report with approximately 800-1000 words on the product, making it as detailed and specific as possible. Your aim is to capture the full essence of the product.
#                   \n \
#                   NOTES ON FORMAT:
#                   Be confident, do not say there is incomplete information, or there is not information. If you can't answer elements from the above, ignore it! Speak as if you are the authority of the subject. If you don't know the answer, don't talk about it. Do not say "I was unable to find information on XYZ". 
#                   """})

#     product_research_3 = product_research_context['output']
#     # if "I couldn't find more information" in product_research_context:
#     #       product_research_1= "Insufficient information. Please write the product description yourself."
#     anvil.server.task_state['result'] = product_research_3


# ####### --------VIDEO SALES SCRIPT --------###################################################

# @anvil.server.callable
# def launch_generate_vsl_script(chosen_product_name, chosen_final_headline, chosen_final_subheadline, chosen_company_profile, chosen_product_research, chosen_avatar, chosen_tone, example_script):
#     print("Launch Generate Video Sales Letter Script Function")
#     current_user = anvil.users.get_user()
#     user_table_name = current_user['user_id']
#     # Get the table for the current user
#     user_table = getattr(app_tables, user_table_name)
#     row = user_table.get(variable='vsl_script')
  
#     # Launch the background task
#     task = anvil.server.launch_background_task('generate_vsl_script', chosen_product_name, chosen_final_headline, chosen_final_subheadline, chosen_company_profile, chosen_product_research, chosen_avatar, chosen_tone,example_script,row)
#     # Return the task ID
#     return task.get_id()

# @anvil.server.background_task
# def generate_vsl_script(chosen_product_name, chosen_final_headline, chosen_final_subheadline, chosen_company_profile, chosen_product_research, chosen_avatar, chosen_tone,example_script,row):
#     # Return the task ID):
#     print("Background task started for generating the Video Sales Letter script")

#     llm_vsl_script = ChatOpenAI(temperature=0.8, model_name='gpt-4', openai_api_key=openai_api_key)

#     vsl_script_template = """You are RussellAI, a highly-evolved version of Russell Brunson, the author and business coach behind "Dotcom Secrets". You are the best scriptwriter on the planet. You are about to launch a brand new video sales letter marketing funnel selling {chosen_product_name}, and you're ready to start write the video sales letter script! This has a very specific format, and requires a lot of context, provided below:
     
#     First, I will provide you with some tips about writing, then I will give you the existing headlines, some context about the company, the ideal customer we're trying to serve, followed by information about the product. Therafter, I will provide you some existing sales scripts (from a parallel industry) that will inform you of style and length. Lastly, I'll request this in an certain order, and provide you with a template to follow.
#     TIPS: This script helps build credibility quickly by answering the problems our avatar faces, provides credibility, explains the product, then gives reasons to act now. It's important to remember that people didn’t come looking for our product… instead we are interrupting them in their daily journey. The only way to get them to stop scrolling online is to grab their attention with an irresistible scripts!

#     HERE IS THE EXISTING HEADLINE and SUBHEADLINE: '{chosen_final_headline}', and '{chosen_final_subheadline}'
#     HERE IS SOME CONTEXT ABOUT THE COMPANY: {chosen_company_profile}
#     HERE IS SOME CONTEXT ABOUT THE PRODUCT: {chosen_product_research}

#     HERE IS THE EXISTING CUSTOMER: {chosen_avatar}
    
#     HERE ARE SOME EXAMPLES OF EXISTING SCRIPTS FROM PARALELL INDUSTRIES. YOU MUST UPDATE IT ACCORDING TO OUR PRODUCT AND COMPANY CONTEXT: {example_script} 

#     TONE: {chosen_tone}

#     HERE IS THE TEMPLATE TO FOLLOW WHEN CREATING THE SCRIPT:
#     Explain The Problem — What problem is our avatar and target market facing? How can we empathize with their challenges? (should be between 90-100 words)
#     Agitate The Problem — What are some examples of that problem? Make that problem visceral for them. Explain why it’s a bigger problem than they think it is and how it’s really going to harm them over the long-run. (should be between 90-100 words)
#     Introduce The Solution — What is your solution to their problem? It's our product, of course! (should be between 90-100 words)
#     Build Credibility — Why should they trust our founder to be the provider of this solution? Use their name. What makes you so great? Telling a story about your own journey can help build credibility. (should be between 90-100 words)
#     Show Proof — How do they know that it’ll actually work? Make up a fictional case-study using ficticious details. This is important to discuss and show proof. (should be between 90-100 words)
#     Explain Exactly What They Get — Explain exactly what the prospect is going to get if they sign up! (should be between 90-100 words)
#     Give Reason To Act Now — Why should they buy right now? Use urgency or scarcity to put the prospect’s foot on the gas.(should be between 90-100 words)
#     Close — Close the sale with a final call-to-action. 

#     The output should be a script, written in the first person from the perspective of the founder that is trying to sell the audience on why their product is the best choice and will make their life easier. The script should not include any subheadings!"""

#     vsl_script_prompt = PromptTemplate(
#         input_variables=["chosen_product_name", "chosen_final_headline", "chosen_final_subheadline", "chosen_company_profile", "chosen_product_research", "chosen_avatar", "chosen_tone","example_script"],
#         template=vsl_script_template
#     )

#     chain_vsl_script = LLMChain(llm=llm_vsl_script, prompt=vsl_script_prompt)
#     vsl_script = chain_vsl_script.run(chosen_product_name=chosen_product_name, chosen_company_profile=chosen_company_profile,chosen_product_research=chosen_product_research,chosen_avatar=chosen_avatar, chosen_tone=chosen_tone,example_script=example_script,chosen_final_headline=chosen_final_headline,chosen_final_subheadline=chosen_final_subheadline
#     )
#     anvil.server.task_state['result'] = vsl_script

# ####### --------VIDEO SALES SCRIPT 4 THEMES --------###################################################

# @anvil.server.callable
# def launch_generate_vsl_themes(chosen_final_headline, chosen_final_subheadline, chosen_product_name, chosen_product_research, chosen_tone,vsl_script,owner):
#     print("Launch Generate VSL Themes Function") 
#     # current_user = anvil.users.get_user()
#     # owner = current_user['email']
#     # Launch the background task
#     task = anvil.server.launch_background_task('generate_vsl_themes',chosen_final_headline,chosen_final_subheadline,chosen_product_name, chosen_product_research, chosen_tone,vsl_script,owner)
#     # Return the task ID
#     return task.get_id()

# @anvil.server.background_task
# def generate_vsl_themes(chosen_final_headline,chosen_final_subheadline, chosen_product_name, chosen_product_research, chosen_tone,vsl_script,row):
#   llm_vsl_themes = ChatOpenAI(temperature=0.8, model_name='gpt-4', openai_api_key=openai_api_key)
#   four_vsl_themes_template = """ You are RussellAI, a highly-evolved version of Russell Brunson, the author and business coach behind "Dotcom Secrets". You are the best scriptwriter on the planet. You are about to launch a brand new video sales letter marketing funnel selling {chosen_product_name}, and you've already generated the sales video letter script, but you now need to extract the four themes from the script and promote them as the industry's dirty secret that will make them millions! These will be captions to screenshots from the video.

#     These extractions help build credibility quickly by addressing the pain points of our customer, provides credibility, explains the product, then gives reasons to act now. It's important to remember that people didn’t come looking for our product… instead we are interrupting them in their daily journey. The only way to get them to stop scrolling online is to grab their attention with an irresistible scripts!
#     First, I will provide you with video's main headline then some context about the product. Therafter, I will provide you with the final script that I need you to summarize and extract themes and reveal the big secrets of our product. Lastly, I'll request this in an certain order, and provide you with a template to follow.
    
#     INGEST THE BELOW INFORMATION WITH SQUARE BRACKETS AS CONTEXT:
#     [
#     EXISTING HEADLINES: {chosen_final_headline}, plus {chosen_final_subheadline}

#     CONTEXT ABOUT THE PRODUCT: {chosen_product_research}

#     HFINAL SCRIPT OF THE VIDEO I NEED YOU TO EXTRACT THE BIG SECRETS FROM: {vsl_script}
    
#     TONE: {chosen_tone}

#     DO NOT INCLUDE ANY SUMMARIZATION OF THE ABOVE POINTS IN THE OUTPUT. I AM ONLY INTERESTED IN THE BELOW OUTPUT:
    
#     ----- FINAL OUTPUT IS BELOW-----HERE IS THE TEMPLATE TO FOLLOW WHEN CREATING THE 4 EXCERPTS

#     "SECRET #1:" 5-7 words of the theme or secret reveal, but in the form of a cheeky and confident headline. Then, provide an exciting sentence about how to be successful in that area, then trail off with an ellipses like this ....
#     "SECRET #2:" 5-7 words of the theme or secret reveal,but in the form of a cheeky and confident headline. Then, provide an exciting sentence about how to be successful in that area, then trail off with an ellipses like this ....
#     "SECRET #3:" 5-7 words of the theme or secret reveal, but in the form of a cheeky and confident headline. Then, provide an exciting sentence about how to be successful in that area, then trail off with an ellipses like this ....
#     "SECRET #4:" a mini-headline that is 5-7 words of what, but  can be next in how they apply these themes. Then, provide a sentence about the magic results they could see..
    
#     For example, the a potential output could look like below.

#     SECRET #1: 'Attribution Agitation' at its Worst: Fed up with not knowing where your sales are coming from? Discover how Funnelytics Performance can clear up the confusion for good...

#     SECRET #2: 'Ad-Cost Anguish' Annihilated: Struggling with soaring ad costs and sub-par results? Learn how to optimize your campaigns and slash ad spend with our innovative platform...

#     SECRET #3: 'Funnel Failure' Flipped Upside Down: Tired of ineffective marketing funnels that just don't deliver? Watch as Funnelytics Performance revamps your funnel strategies and turns them into massive growth engines...

#     SECRET #4: 'Scaling Struggles' Solved: Wondering how to grow your business without breaking the bank? Witness the magic as Funnelytics Performance helps you unlock unprecedented growth and skyrocket your success...'
#     ]

#     NOTE: The Final Output will be just Secret 1 through 4. NOT EVEN QUOTE MARKS "". Nothing else!
#     THAT'S IT...NOW GO AND CREATE THE 4 EXCERPTS!
#     --

#     SECRET #1:...
#     SECRET #2:...
#     SECRET #3:...
#     SECRET #4:..
#     """

#   vsl_themes_prompt = PromptTemplate(
#       input_variables=["chosen_final_headline","chosen_final_subheadline","chosen_product_name", "chosen_product_research", "chosen_tone","vsl_script"], 
#       template=four_vsl_themes_template
#   )

#   chain_vsl_themes = LLMChain(llm=llm_vsl_themes, prompt=vsl_themes_prompt)
#   vsl_themes_generator = chain_vsl_themes.run(chosen_final_headline=chosen_final_headline, chosen_final_subheadline=chosen_final_subheadline,chosen_product_name=chosen_product_name, chosen_product_research=chosen_product_research, chosen_tone=chosen_tone,vsl_script=vsl_script)
#   print("Here are the 4 excerpts", vsl_themes_generator) 
#   vsl_themes = vsl_themes_generator.split("\n")

#   # Initialize an empty list
#   all_vsl_themes = []

#  # Loop over each line
#   for vsl_theme in vsl_themes:
#       # Ignore empty lines
#       if not vsl_theme.strip():
#           continue

#       # Append the headline to the list
#       all_vsl_themes.append(vsl_theme.strip())

#   # Convert the list to a JSON string
#   all_vsl_themes_json = json.dumps(all_vsl_themes)

#   row['variable_value'] = all_vsl_themes_json
#   row.update()

#   # Return the resulting JSON string
#   anvil.server.task_state['result'] = all_vsl_themes_json