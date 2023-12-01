from openai import OpenAI

class SimpleClient:
    def __init__(self, key):
        self.client = OpenAI(api_key=key)
    
    def get(self, layout, query):
        completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Provide proper elastic search query with provided fields. Do not explain, just a query"
                },
                {
                    "role": "user",
                    "content": f"with the doc fields {layout}, search all docs whose {query}"
                }
            ],
            model="gpt-3.5-turbo"
        )
    
        return completion.choices[0].message.content