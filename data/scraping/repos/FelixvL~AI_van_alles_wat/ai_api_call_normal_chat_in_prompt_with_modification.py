import openai
openai.api_key = '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'


prompt = input("waarover wil je een reactie van ChatGPT")


response = openai.Completion.create(
  engine='text-davinci-003',  # Specify the language model you want to use
  prompt="what are the famous siteseeings of:  "+prompt,
  max_tokens=100  # Adjust the desired length of the completion
)


generated_text = response.choices[0].text.strip()


print(generated_text)