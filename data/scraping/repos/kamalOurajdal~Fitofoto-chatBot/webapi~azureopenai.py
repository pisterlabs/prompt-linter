import openai
openai.api_type = "azure"
openai.api_base = "https://allrythm.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key =  "65b80e5fd44c40bea8aa32e3e87b02ec"

def sendgpt(prompt):
    response = openai.Completion.create(
      engine="Greenius",

      prompt=prompt,
      temperature=0.7,
      max_tokens=64,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["\n"]
    )
    return response.choices[0].text