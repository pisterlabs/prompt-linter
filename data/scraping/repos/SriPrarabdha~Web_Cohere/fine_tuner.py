import cohere
from data_extractor import loader

co = cohere.Client('xCRrVzZjuPM5HN6WFKM1eykBBwHMezrMhaQ0AaD7')

data = loader.load()

dataset= []
for i in range(len(data)):
    res = co.summarize(
    text=data[i].page_content,
    model="summarize-xlarge",
    length="long",
    temperature=0.3,
    additional_command="summarize it in form of 5 question that can be a high quality prompt to a Large Language model")

    prompts = res.summary.splitlines()
    for prompt in prompts:
        prompt = prompt[2:]
        dataset.append({
            "prompt" : data[i].metadata["title"] + data[i].metadata["description"] + prompt,
            "completion" : data[i].page_content
        })

print(dataset)