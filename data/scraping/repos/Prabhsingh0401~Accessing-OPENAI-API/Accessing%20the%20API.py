import openai

openai.api_key = "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"

# Use openai.ChatCompletion to use the Chat Model of openai 
# Images are also there in repository for the working and usage of Models

response = openai.Completion.create(
    model = "text-davinci-003",
    prompt= "What is Openai API"
)

print(response["choices"][0]["text"])          # The Output is in form of JSON File and it is nested in lists so to get only text from those list we print the only string which is necessary 