import openai
import os

def result_to_latex(answer):
  openai_key = os.environ['OpenAI Key']
  openai.api_key = openai_key
  response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = [
        {"role": "system", "content": "You are the world's best machine that does nothing but convert strings into LaTeX flawlessly. You do not make mistakes."},
        {"role": "user", "content": "Convert this following string into latex, with the mathematical portions of this string correctly converted into the proper mathematical formatting in LaTeX. Do not output anything other than the LaTeX code in a code window. I want the resulting LaTeX code as flawless as possible, so that I can copy and paste it into a section of an existing LaTeX code body and it will compile without problems while maintaining the perfect formatting. {}".format(answer)}
    ]
  )
  return response['choices'][0]['message']['content']