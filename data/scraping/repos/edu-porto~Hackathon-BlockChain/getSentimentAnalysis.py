import openai

class GetSentimentAnalysis():
    def __init__(self):
        pass


    def sentimentalAnalysis(self, summaries):
        sentiment = []
        for summary in summaries:
            response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Classify the sentiment in this news input : {summary}",
            temperature=0,
            max_tokens=1500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
            )
            
            sentiment.append(response["choices"][0]["text"])

        return sentiment


