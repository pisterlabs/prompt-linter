import openai
import io

#class for lyrics generation using chatGPT
class lyrics_generator:
    def __init__(self) -> None:
        api_key = "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"
        openai.api_key = api_key

    def send_request(prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message["content"]
    
    def removeCaptions(text):
        stopWords = ['Verse', 'Chorus', 'Bridge', 'Outro']
        response = ""
        for line in io.StringIO(text):
            if not any(word in line for word in stopWords):
                response = response + line
        return response
