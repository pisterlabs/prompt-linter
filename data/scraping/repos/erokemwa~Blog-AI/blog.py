import os
import openai
import sys

openai.api_key = "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"
if (len(sys.argv) != 2):
        print("Error: need the title please")
        exit()
title = sys.argv[1]
blog_post = title + "\n"
print(title + "\n")
prompt = "I want to write a blog post about '" + title + "'. Give a list of 5 s\
ections in a numbered bullet point format about this blog post."
synopsis = openai.Completion.create(
  model = "text-davinci-003",
  prompt = prompt,
  max_tokens = 1000,
  temperature = 0.7
)
synopsis = synopsis.choices[0].text
synopsis = synopsis.strip()

lines = synopsis.strip().splitlines()
for section in lines:
    blog_post = blog_post + section + "\n"
    print(section + "\n")
    prompt = "I am writing a blog post with the title '" + title + "'.\n\nThe l\
ist of sections of this blog post with the following:\n" + synopsis + "\n\nWrite \
the section '" + section + "' in a deatiled and complete way, in 500 word mini\
mum."
    section_paragraph = openai.Completion.create(
        model = "text-davinci-003",
        prompt = prompt,
        max_tokens = 750,
        temperature = 0.7
    )
    section_paragraph = section_paragraph.choices[0].text
    section_paragraph = section_paragraph.strip()
    print(section_paragraph + "\n")
    blog_post = blog_post + section_paragraph + "\n"

print(blog_post)
filename = title + "txt"
with open(filename, "w") as f:
    f.write(blog_post)
