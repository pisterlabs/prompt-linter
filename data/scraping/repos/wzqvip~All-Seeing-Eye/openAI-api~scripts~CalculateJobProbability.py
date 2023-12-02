import openai
import os
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

user_description = "I have a degree in Computer Science, 2 years of experience in software development, and I'm proficient in Python and Java. I want to apply for the position of Senior Software Engineer at Google."
# user_description= "I'm graduated from elementary school but I have doctor's degree. I want a job as Presendent."
user_description = "I have ten years of driving experience. I want to be a texi driver to earn a live. Will i success?"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    messages=[
        {"role": "system", "content": "You are a helpful career planning assistant. You will tell user how much he could be accpted(in percentage). And tell him what else he can do if the percentage is below 50%(at least 3)."},
        {"role" : "user",  "content": "This is an example output,please follow this format: Calculated success rate of Microsoft is 10%. Based on your description of not having any computer knowledge, the chances of being accepted at Microsoft are quite low.\n\nHowever, I can suggest some alternative career options with a higher probability of acceptance:\n\n1. Sales Representative: Success rate 75%\n2. Marketing Coordinator: Success rate 80%\n3. Administrative Assistant: Success rate 85%\n4. Customer Service Representative: Success rate 90%\n\nThese are just a few options to consider based on your current skillset. It's important to remember that gaining computer knowledge and skills can significantly increase your chances of being accepted into the tech industry."},
        {"role": "user", "content": "I want to apply for a job, can you tell me how much possibility I could be accepted. And what else can I do ? You can directly reply to my questions without any explanation."},
        {"role": "assistant", "content": "Sure. Please Describe yourself and the job you are going to acquire. I will give you the chance you will be accepted and provide you with the jobs you also could be suit for.."},
        {"role": "user", "content": user_description},
        {'role': 'system', 'content':" Now please output the percentage that the user will be accpeted by the company, in the format of 'Calculated success rate of $company_name is $percentage% . If the percentage is below 50%, please offer him alternative choices."}
    ]

)



# messages=[
#     {"role": "system", "content": "You are a career planning assistant."},
#     {"role": "user", "content": "I have a degree in Computer Science, 2 years of experience in software development, and I'm proficient in Python and Java. I want to apply for the position of Senior Software Engineer at Google."}
# ]

print(response['choices'][0]['message']['content'])

with open('probability.json', 'w') as f:
    json.dump(response, f)
