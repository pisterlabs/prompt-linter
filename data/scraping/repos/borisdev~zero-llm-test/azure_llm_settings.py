from langchain.llms import AzureOpenAI

llm = AzureOpenAI(
    deployment_name="boris-text-davinci-003",
    model_name="text-davinci-003",
    temperature=0,
    max_tokens=512,
)
