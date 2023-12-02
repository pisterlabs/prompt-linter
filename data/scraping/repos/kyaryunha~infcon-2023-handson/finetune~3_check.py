import openai

from api_key import api_key
openai.api_key = api_key

findtune_id = "ft-"
print(openai.FineTune.list_events(findtune_id))

