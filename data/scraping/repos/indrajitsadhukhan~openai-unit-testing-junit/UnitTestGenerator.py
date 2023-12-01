import openai
input_file=input("Type the filename: ")
# import fileinput
import fileinput

code_txt=""
# Using fileinput.input() method
for line in fileinput.input(files = input_file):
    code_txt+=line
    code_txt+='\n'

openai.api_key=""


question = "Write Junit test for this code"
prompt = code_txt + question



# send the dataframe to the ChatGPT model
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=(prompt),
    temperature=0,
    max_tokens=2000,
    top_p=1,
    frequency_penalty=1,
    presence_penalty=1
)
print(response["choices"][0]["text"])
