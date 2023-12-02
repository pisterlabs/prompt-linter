import guidance
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

valid_dish = ["Pizza", "Noodles", "Pho"]

guidance.llm = guidance.llms.OpenAI(model="text-davinci-003", api_key=OPENAI_API_KEY)

# define the prompt
order_maker = guidance("""The following is a order in JSON format.
```json
{
    "name": "{{name}}",
    "age": {{gen 'age' pattern='[0-9]+' stop=','}},
    "delivery": "{{#select 'delivery'}}Yes{{or}}No{{/select}}",
    "order": "{{select 'order' options=valid_dish}}",
    "amount": {{gen 'amount' pattern='[0-9]+' stop=','}}
}```""")

# generate a character
order_maker(
    name="Alex",
    valid_dish=valid_dish
)