import openai

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
openai.api_key = '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'

def generate_story(user_input):
    prompt = f"Once upon a time, {user_input}."

    try:
        response = openai.Completion.create(
            engine="text-davinci-002",  # Replace with the GPT-3.5 engine name
            prompt=prompt,
            temperature=0.8,
            max_tokens=400
        )

        story = response.choices[0].text.strip()
        return story
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    user_input = input("Enter a starting phrase for the story: ")
    generated_story = generate_story(user_input)
    print("\nGenerated Story:\n")
    print(generated_story)
