import openai
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

def ClassifyTextUsingChatGPT(text_to_classify,
                             categories,
                             openai_api_key,
                             my_prompt_template="""You will be provided with text delimited with triple backticks \
                                Classify each text into a category. \
                                Provide your output in json format with the keys: "text" and "category". \
                                Categories: {categories}. \
                                text: ```{text_to_classify}```""",
                             print_api_cost=True,
                             temperature=0.0,
                             chat_model_name="gpt-3.5-turbo"):  
    # Set the chat prompt template
    prompt_template = ChatPromptTemplate.from_template(my_prompt_template)
    
    # Set the messages
    messages = prompt_template.format_messages(
        text_to_classify=text_to_classify,
        categories=categories
    )
    
    # Set the model name and temperature
    chatgpt_model = ChatOpenAI(
        temperature=0.0, 
        model_name=chat_model_name,
        openai_api_key=openai_api_key
    )
    
    # Send the prompt to the OpenAI API 
    response = chatgpt_model(messages)
    
    # Return the response
    return(response.content)


# # Test the function
# # Read in OpenAI API key
# my_openai_api_key = open("C:/Users/oneno/OneDrive/Desktop/OpenAI key.txt", "r").read()
# # Set the categories
# possible_categories = "Billing, Technical Support, \
# Account Management, or General Inquiry."
# # Set the user message
# user_message = f"""
# I want you to delete my profile and all of my user data
# """
# # Classify the text
# response = ClassifyTextUsingChatGPT(
#     text_to_classify=user_message,
#     categories=possible_categories,
#     openai_api_key=my_openai_api_key,
#     print_api_cost=True,
#     temperature=0.0
# )
# print(response)
