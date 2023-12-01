import openai

# Open API key
openai.api_key = "your_key_here"

class PoemCreator:
    def __init__(self):
        pass

    # To use OpenAI ChatGPT to generate valid haiku from processed lyrics
    def generate_haiku(self, your_lyrics):
        prompt = "Create a silly, formal, and whimsical poem that cannot exceed 280 characters. " \
                 "Make the poem " \
                 "capture the meaning of the following lyrics, and include words within the lyrics when you" \
                 "write the poem. Please do not exceed 10 lines of writing and do not use offensive slurs." \
                 "Here are the lyrics: \n" + your_lyrics
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a kind and sweet poet assistant "
                                              "who wants to make others laugh."},
                {"role": "user", "content": prompt}
            ],
            n=1,
            max_tokens=200,
            temperature=0.8
        )
        haiku = response.choices[0].message['content'].strip()
        return haiku
