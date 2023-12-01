import openai

# Replace with your OpenAI API key
api_key = "YOUR_API_KEY"

def generate_captions(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",  # You can choose a different engine if needed
            prompt=prompt,
            max_tokens=50,  # Adjust the max_tokens as needed for desired caption length
            api_key=api_key,
        )

        if response.choices:
            captions = [choice.text.strip() for choice in response.choices]
            return captions
        else:
            print("Failed to generate captions. Please check your API key and prompt.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    while True:
        topic = input("Enter a topic to generate captions (e.g., love, anime, bike, car, friends): ")
        prompt = f"Generate captions on the topic of {topic}"
        captions = generate_captions(prompt)
        
        if captions:
            print("\nHere are three captions on the topic:")
            for i, caption in enumerate(captions[:3], 1):
                print(f"{i}. {caption}")
            
            more_captions = input("\nDo you want more captions on this topic or captions on another topic? (type 'more' or 'exit'): ").lower()
            if more_captions != 'more':
                break
        else:
            print("No captions generated for the given topic.")

if __name__ == "__main__":
    main()
