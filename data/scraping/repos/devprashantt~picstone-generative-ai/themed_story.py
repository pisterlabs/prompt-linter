import openai


def generate_themed_story(
    theme
):
    # Create a prompt with specific instructions for ChatGPT
    prompt = f"Generate an intriguing story based on the {theme} theme. The story should include suspenseful events, unexpected twists, and engaging characters. Ensure that the story maintains a sense of {theme} throughout, keeping the user captivated until the resolution. Consider incorporating elements such as hidden clues, enigmatic settings, and characters with ambiguous motives. The generated story should be immersive and evoke a sense of curiosity. Keep the user engaged by introducing new elements that deepen the mystery and lead to a satisfying conclusion. Be creative and make the story dynamic and compelling."

    try:
        # Generate a story/poem using ChatGPT
        response = openai.Completion.create(
            engine="text-davinci-003",
            temperature=0.7,  # Adjust temperature for creativity
            max_tokens=1000,  # Adjust max_tokens for desired length
            prompt=prompt,
            n=1             # Ensure only one response is generated
        )

        return response.choices[0].text
    except Exception as e:
        print(f"Error generating poem/story from Server: {str(e)}")
        return None
