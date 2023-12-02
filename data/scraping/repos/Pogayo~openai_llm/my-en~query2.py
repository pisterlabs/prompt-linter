
# strategy
import os
import openai

openai.api_key = "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"

# names_list = [name.strip() for name in open("../machine-translation/english_names.tsv").readlines()]
names_list = [
"အော့ဝနၣ",
"ဆမူအေးလၣ",
"ဂျေဘောၣ",
"အေရာၣ",
"အိဒနၣ",
"ဂျော့ဆေးၣ",
"ဂျိုးပါးၣ",
"ဒေဝေဝၣ"
]
# out_file = open("names.my", "a")
# print(instruct)
# examples = ["these, are, five, oranges",
#             "these, are, five, oranges",
#             "these, are, 5, oranges",
#             "these, are, 16, oranges",
#             "these, are, four, oranges"
#             "these, are, sweet, oranges"]

# prompt1 = instruct + "Sentence 1: " + examples[0] + "\n"
for i in range(1):
    # prompt = prompt1 + "Sentence 2: " + examples[i] + "\n\n Answer: "
    no = i*20
    names = "\n".join(names_list)
    instruct = "Here is a list of names in Burmese: " + names + \
    " For each name, create unique and well-constructed sentences in Burmese and their English translations using those names." \
    " Try to use varied sentence structures and meanings to showcase your understanding of language. " \
    "For example, one sentence could be a simple statement, another a complex compound sentence, another a question, and the fourth an exclamation."

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=instruct,
        temperature=0.14,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    result = response['choices'][0]['text']
    print(result)
    # out_file.write(result.strip()+"\n")
