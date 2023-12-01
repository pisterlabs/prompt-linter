# Create aoai class to connect to Azure OpenAI API and have a run function to send a prompt and get a response

class aoai:
    import os
    import openai
    import dotenv

    dotenv.load_dotenv()
    openai.api_type = "azure"
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_version = "2023-03-15-preview"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def __init__(self, engine: str=os.getenv("ENGINE") ):
        self.engine = engine

    def run(self, prompt, temperature: float=0.5, max_tokens: int=500, 
                 top_p: float=0.5, 
                 frequency_penalty: float=0.0, 
                 presence_penalty: float=0.0,
                 stop: list=[],
                 isRaw: bool=False):
        response = self.openai.ChatCompletion.create(
            engine=self.engine,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        )

        if isRaw:
            return response
        else:
            return response.choices[0].message['content']