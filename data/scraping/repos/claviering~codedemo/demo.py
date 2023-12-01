import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://closeai.deno.dev/v1"

end_indicator_string = "\n\n###\n\n"

response = openai.Completion.create(
  model="curie:ft-personal:text-curie-wenai-001-2023-02-17-09-32-19",
  prompt="我肚子饿了。" + end_indicator_string,
  stop=["###"],
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
print(response.usage)
choices = response.choices
for choice in choices:
    text = choice.text.encode('utf-8').decode('unicode_escape')
    print(text)