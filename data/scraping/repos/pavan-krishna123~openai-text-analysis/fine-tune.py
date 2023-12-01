import openai

with open('openaiapikey.txt', 'r') as infile:
    open_ai_api_key = infile.read()

openai.api_key = open_ai_api_key


def file_upload(filename, purpose='fine-tune'):
    resp = openai.File.create(purpose=purpose, file=open(filename))
    print(resp)
    return resp


def finetune_model(fileid, suffix, model='davinci'):
    resp = openai.FineTune.create(
        training_file=fileid,
        model=model,
        suffix=suffix
    )
    print(resp)


resp = file_upload('Sentiment.jsonl')  # Replace with the name of your JSON Lines file
finetune_model(resp['id'], 'sentiment_classifier', 'davinci')
