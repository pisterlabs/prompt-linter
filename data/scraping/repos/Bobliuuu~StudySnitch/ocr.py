import easyocr
import openai

openai.api_key = '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'

reader = easyocr.Reader(['en']) 
result = reader.readtext('photos/linear algebra.png', detail = 0)
text = " ".join(result)

prompt = "Given this text, return whether it is related or not to linear algebra. Only return yes or no. Do not return anything else other than yes or no."

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages = [
        {'role': 'user', 'content': prompt}
    ],
    max_tokens=1024,
    temperature=0,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=['#'])['choices'][0]["message"]

print(response)


