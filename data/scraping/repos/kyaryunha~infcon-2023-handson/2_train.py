import openai

from api_key import api_key
openai.api_key = api_key


def train():
    training_file = "file-"

    finetune = openai.FineTune.create(
        training_file=training_file,
        model="davinci",
    )
    return finetune["id"]

# findtune_id = train()
findtune_id = "ft-"
print(openai.FineTune.list_events(findtune_id))

# print(openai.FineTune.list())

