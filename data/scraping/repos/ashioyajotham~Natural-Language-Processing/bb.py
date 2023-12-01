# The first step is to import the OpenAI API
import openai

# The second step is to set the API key
api_key = ""

# The third step is to set the OpenAI API key
openai.api_key = api_key

# The fourth step is to create a function that will generate the chatbot's response
def chatbot_response(question):
    prompt = f"Question: {question}\nAnswer:"
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n"]
    )
    return response.choices[0].text

# The fifth step is to create a function that will ask the user a question and return the chatbot's response
def ask_question():
    question = input("Ask a question about the inflation in Poland: ")
    answer = chatbot_response(question)
    print("Answer: " + answer)

# The sixth step is to ask the user a question and return the chatbot's response
ask_question()

# The seventh step is to save the chatbot's response to a csv file
df.to_csv("chatbot_response.csv", index=False)

# The eighth step is to save the chatbot's response to a json file
df.to_json("chatbot_response.json", orient="records")