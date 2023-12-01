import openai

class GetGPTSummaries():
    def __init__(self, api_key):
        self.api_key = api_key


    def getGPTSummaries(self, news_text):
        openai.api_key = self.api_key
        summaries = []
        for text in news_text:
            completition = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = [{"role": "user", "content":f"Sumary this text and get the key insights :{text}" }],
            max_tokens = 1024,
            temperature = 0.8)
            summaries.append(completition["choices"][0]["message"]["content"])
            
        return summaries

