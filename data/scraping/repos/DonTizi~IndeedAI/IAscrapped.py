from openai import OpenAI

class JobDescriptionWriter:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def compose_presentation_letter(self, description):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an eloquent writer with skills to write excellent and persuasive texts."},
                {"role": "user", "content": f"Compose a letter of presentation for this company my interest to the job position to get hire. Here is the description of the job:\n{description}"}
            ]
        )
        return completion.choices[0].message.content