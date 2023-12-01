import environ
import os
import openai

env = environ.Env()

environ.Env.read_env()

# from pkg_resources import working_set

# installed_packages = ("%s==%s" % (i.key, i.version) for i in working_set)
# installed_packages_list = sorted(installed_packages)

# print(installed_packages_list)

def return_haiku():
    
    openai.api_key = env("OPENAI_API_KEY")

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Hello Chat, please generate me a humorous original haiku in lowercase."}
    ]
    )

    haiku = completion.choices[0].message.content.split('\n')

    line_1, line_2, line_3 = haiku[0], haiku[1], haiku[2]

    return [line_1, line_2, line_3]
    # return ["this is my haiku,", "it is the best I have done,", "thanks for listening"]