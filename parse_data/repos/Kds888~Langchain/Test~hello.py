# import os
# from langchain import PromptTemplate
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# from third_parties.linkedin import get_info_from_linkedin
# from agents.linkedin_agent import lookup_linkedin



# name=input("Enter The Name of the person you want to search on LinkedIn: --  ")

# get_linked_in_url = lookup_linkedin(name=name)

# get_linked_in_id = get_linked_in_url.split('/')[-1] 



# information = get_info_from_linkedin(profile_id=get_linked_in_id)
# print(information)

# # information will be changing from time to time.
# summary_template = """
# Given the LINKEDIN information {information} about a person I want to you to create: 
# 1.Short Summary
# 2. Two interesting facts about them 
# """

# summary_template_prompt = PromptTemplate(
#     template=summary_template, input_variables=["information"]
# )
# # As the prompt template makes use of the word information therefore we need the information name ourselves

# llm = ChatOpenAI(
#     temperature=0, model_name="gpt-3.5-turbo"
# )  # Temperature will decide how creative the model will be

# chain = LLMChain(llm=llm, prompt=summary_template_prompt)

# print(chain.run(information=information))

















# ##############################################################################################################################################
# # import openai # Image generation using Dall3-2 but it is not working for some reasons.


# # response = openai.Image.create_variation(
# #   image=open("1.png", "rb"),
# #   n=1,
# #   size="1024x1024",
# #   )
# # image_url = response['data'][0]['url']
# # print(image_url)
