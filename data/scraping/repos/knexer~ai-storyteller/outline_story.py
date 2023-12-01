import openai

class Outliner:
    def __init__(self, conditioning_info, premise):
        self.conditioning_info = conditioning_info
        self.premise = premise

    def outline_prompt(self):
        return f"""You are an AI storyteller who specializes in illustrated books for young children. You are fleshing out a story premise for a client today.
The client's requirements are:
{self.conditioning_info}

The story premise is:
{self.premise}

First, expand that quick premise into a more detailed form, including:
- What characters are involved in the story? What are their motivations? How might they interact with and relate to one another?
- What settings and environments will the story take place in? What makes them unique? How do they relate to the premise and the characters?
- What are the main themes and messages in the story?

Then, given that information, write an outline for the story.
Follow a classic narrative structure: exposition, rising action, climax, falling action, and resolution.
Ensure that the story has a clear message and a logical progression.
"""

    def outline(self, n=1):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": self.outline_prompt()},
            ],
            n=n,
            temperature=1,
        )

        return [choice.message.content for choice in response.choices]
