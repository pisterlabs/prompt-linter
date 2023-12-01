import openai

GPT3_MODELS = {
    'davinci': 'text-davinci-003',
    'curie': 'text-curie-001',
    'babbage': 'text-babbage-001',
    'ada': 'text-ada-001'
}
        
CODEX_MODELS = {
    'davinci': 'code-davinci-002',
    'cushman': 'code-cushman-001'
}

class OpenAIPlayground:    
    def __init__(self, api_key):
        self.openai = openai
        self.openai.api_key = api_key
        self._conversation_log = ''
    
    def grammar_checker(self, prompt, model=GPT3_MODELS['davinci'], temperature=0.0, max_tokens=1000, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
        response = self.openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=1
        )
        result = {
            'id': response.id,
            'created': response.created,
            'model': response.model,
            'completion_tokens': response.usage.completion_tokens,
            'prompt_tokens': response.usage.prompt_tokens,
            'total_tokens': response.usage.total_tokens,
            'outputs': response.choices[0].text,
            'status': response.choices[0].finish_reason
        }
        return result