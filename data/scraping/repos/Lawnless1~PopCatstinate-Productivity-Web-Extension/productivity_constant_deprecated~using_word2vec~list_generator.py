import os
import openai
import json

openai.api_key = "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"

script_dir = os.path.dirname(__file__)
read_filepath = os.path.join(script_dir, 'top_1000_websites.txt')
with open(read_filepath, 'r') as file:
    contents = file.read()


list_contents = contents.replace(" ", "")
list_contents = contents.split("\n")

domain_list = []
for website in list_contents:
     base_domain = website.split('.')[0]
     if base_domain not in domain_list:
        domain_list += [base_domain]

with open("/home/kc/Desktop/shell-lesson-data/mcgill-mais-hacks-hackathon-2023/server/productivity_constant/using_word2vec/top_1000_websites_easy_to_access.txt", 'a+') as file:
    file.write(str(domain_list))
