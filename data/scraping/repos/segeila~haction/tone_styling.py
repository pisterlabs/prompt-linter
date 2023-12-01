import openai 

openai.api_key = "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"

def tone_styling(text, style):
    if style == "George Carlin":
        query = f"Rewrite the following text using {style} style: {text}"
    elif style == "Atul Gawande - Short":
        style = "Atul Gawande"
        query = f"Rewrite the following text using style of {style}. Make it as concise as possible, but keep the main points : {text}"
    elif style == "Donald Trump - Short":
        style = "Donald Trump"
        query = f"Summarize the text. Make it as concise as possible, but keep the main points : {text}. The re-write the summary using style of {style}. "
    response =  openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful research assistant."},
                        {"role": "user", "content": query},
                            ],
                            )
    return response["choices"][0]["message"]["content"]