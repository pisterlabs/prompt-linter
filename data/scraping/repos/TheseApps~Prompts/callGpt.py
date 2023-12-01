
from openai import OpenAI

client = OpenAI(
  organization='org-o9XRIbvuhIsbfxA9Q98bxQan',
)

print('start')

# instead; output the response to the console
print(client.models.list())


print('end')