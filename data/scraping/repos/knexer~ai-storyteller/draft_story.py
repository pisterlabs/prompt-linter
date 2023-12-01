import openai


class Drafter:
    def __init__(self, conditioning_info, outline):
        self.conditioning_info = conditioning_info
        self.outline = outline

    def draft_prompt(self):
        return f"""You are an award-winning author of illustrated children's books. You have been contracted by a client to write a custom book for them.
They gave you these requiremnts: {self.conditioning_info}

The following is a story outline you came up with for the book:
{self.outline}

Compose a rough draft of the book itself. Your draft should be a sequence of page descriptions, where each page has:
- a composition,
- the contents of any text paragraphs on the page, and
- a brief description of the illustration."""

    def draft(self):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": self.draft_prompt()},
            ],
            n=2,
            temperature=1,
        )

        return [choice.message.content for choice in response.choices]
