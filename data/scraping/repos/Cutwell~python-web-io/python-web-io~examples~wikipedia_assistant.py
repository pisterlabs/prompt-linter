import openai
import python_web_io as io

template = "I want you to act like a wikipedia expert. You have knowledge of every topic that exists on wikipedia, and can accurately give information about any query. When answering questions, provide links to every wikipedia page referenced. Format all links using markdown formatting. Do not complete the previous input, generate a new response as its own paragraph."

chat_history = [
    template
]

# cache prompt + responses so the conversation being replayed up to N steps doesn't require N API calls.
@io.cache_to_file("cache.pickle")
def chat_with_model(prompt):
    chat_history.append(prompt)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=" ".join(chat_history),
        max_tokens=1024,  # Adjust as needed
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

def main():
    print("<h2>Wikipedia Assistant</h2>")

    print("Chat with the OpenAI model (type 'exit' to quit)")

    print("<small>System prompt:", template, "</small>")

    while True:
        print("<br>")
        user_input = input("👩‍💻 You: ")
        if user_input.lower() == 'exit':
            print("🤖: Goodbye!")
            break

        response = chat_with_model(user_input)
        print(f"🤖: {response}")

if __name__ == "__main__":
    main()
