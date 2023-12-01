import os
import openai
import wandb

openai.api_key = ""#key deleted before uploading to github repo

run = wandb.init(project='GPT-3 in Python')
prediction_table = wandb.Table(columns=["prompt", "completion"])

gpt_prompt = "Make a list of astronomical observatories:"
#gpt_prompt = "Please list the steps to remove an item from a box in iManage Records Manager???"
#gpt_prompt = "Respond in a pirate dialect to the following question, Is iManage Records Manager vulnerable to log4j???"
#gpt_prompt = "The trend toward lower rents may seem surprising given that some communities in New York are bemoaning the loss of favorite local businesses to high rents. But, despite the recent softening, for many of these retailers there's still been too big a jump from the rental rates of the late 1970s, when their leases were signed. Certainly, the recent drop in prices doesn't mean Manhattan comes cheap. Human-supplied input : question: Manhattan comes cheap. true, false, or neither?"

response = openai.Completion.create(
  engine="davinci:ft-personal-2022-09-06-18-40-06",
  prompt=gpt_prompt,
  temperature=0.83,
  max_tokens=30,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop="###"
)


print(response['choices'][0]['text'])


prediction_table.add_data(gpt_prompt,response['choices'][0]['text'])

wandb.log({'predictions': prediction_table})
wandb.finish()
