import openai

def GetSentimentScore(text):
    # Use OpenAI to analyze the sentiment of the text
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt = f"Please analyze the sentiment of the following text:{text}",
        temperature=0,
        max_tokens=128,
        n=1,
        stop=None,
        timeout=10,
    )
    # Extract the sentiment from the API response
    #openai_sentiment = response.choices[0].text.strip().replace("The sentiment of the text is ", "").rstrip('.')
   


    return response