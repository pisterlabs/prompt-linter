from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import json
import config.api as config
import config.prompt as prompt

def text_to_link(text):
    """Convert text to link"""

    openai_api_key = config.get_open_api_key()

    chat = ChatOpenAIchat = ChatOpenAI(
        temperature=1, openai_api_key=openai_api_key)
    
    print(prompt.get_prompt())

    # get response from openai
    ans = chat(
        [
            SystemMessage(content=prompt.get_prompt()),
            HumanMessage(
                content=text)
        ]
    )

    print(ans)

    product_information = json.loads(ans.content)

    if product_information['lower_price'] == '0' or product_information['lower_price'] == 0 or product_information['lower_price'] == '':
        product_information['lower_price'] = None
    
    if product_information['upper_price'] == '0' or product_information['upper_price'] == 0 or product_information['upper_price'] == '':
        product_information['upper_price'] = None

    print(product_information)

    queries = ""

    if product_information['search_string'] is not None:
        queries += "k="+product_information['search_string']
    
    if product_information['lower_price'] is not None and product_information['upper_price'] is not None:
        queries += "&rh=p_36%3A" + str(product_information['lower_price']) + "00-" + str(product_information['upper_price']) + "00"
    
    elif product_information['lower_price'] is not None and product_information['upper_price'] is None:
        queries += "&rh=p_36%3A" + str(product_information['lower_price']) + "00-"
    
    elif product_information['lower_price'] is None and product_information['upper_price'] is not None:
        queries += "&rh=p_36%3A-" + str(product_information['upper_price']) + "00"
    
    if product_information['company_name'] is not None and (product_information['lower_price'] is not None or product_information['upper_price'] is not None) :
        queries += ",p_89%3A" + product_information['company_name']
    
    elif product_information['company_name'] is not None and (product_information['lower_price'] is None or product_information['upper_price'] is None):
        queries += "&rh=p_89%3A" + product_information['company_name']
    
    amazon_link = config.get_api_url() + "s?" + queries

    return amazon_link , product_information['sort_by']
