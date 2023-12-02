import openai
import os
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

user_description = "I'm an experienced programmer, software developer. I have been worked at Google, Microsoft. I want to apply for a job as senior software developer at Amazon."
# user_description= "I'm graduated from elementary school but I have doctor's degree. I want a job as Presendent."


response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    messages=[
        {"role": "system", "content": "You are a helpful career planning assistant. You will tell user how he could reach his goal. And tell him what else he can do."},
        # {"role" : "user",  "content": "This is an example output,please follow this format: Calculated success rate of Microsoft is 10%. Based on your description of not having any computer knowledge, the chances of being accepted at Microsoft are quite low.\n\nHowever, I can suggest some alternative career options with a higher probability of acceptance:\n\n1. Sales Representative: Success rate 75%\n2. Marketing Coordinator: Success rate 80%\n3. Administrative Assistant: Success rate 85%\n4. Customer Service Representative: Success rate 90%\n\nThese are just a few options to consider based on your current skillset. It's important to remember that gaining computer knowledge and skills can significantly increase your chances of being accepted into the tech industry."},
        {"role": "user", "content": "I want to apply for a job, can you tell me how much possibility I cloud be accepted. And what can I do to acquire the job? You can directly reply to my questions without any explanation."},
        {"role": "assistant", "content": "Sure. Please Describe yourself and the job you are going to acquire. I will give you the chance you will be accepted and provide you help and also offer you jobs you also could be suit for.."},
        {"role": "user", "content": user_description},
        {'role': 'system', 'content': " Now please output the percentage that the user could reach his goal in the format of 'Calculated success rate of $company_name is $percentage% . If the percentage is above 50%, give him instructions and suggestions. If the percentage is below 50%, please offer him alternative choices."}
    ]
)


# messages=[
#     {"role": "system", "content": "You are a career planning assistant."},
#     {"role": "user", "content": "I have a degree in Computer Science, 2 years of experience in software development, and I'm proficient in Python and Java. I want to apply for the position of Senior Software Engineer at Google."}
# ]

print(response['choices'][0]['message']['content'])

with open('instruction.json', 'w') as f:
    json.dump(response, f)
