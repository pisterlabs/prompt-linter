def ask(query):
    import openai
    import json
    

   
        
    openai.api_key = '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'
    
    response = openai.Completion.create(
        engine = "text-davinci-003",
        prompt = query,
        temperature = 0.5,
        max_tokens = 1000,
        top_p = 1.0,
        frequency_penalty = 0.0,
        presence_penalty = 0.0,
        
    )
    print(response['choices'][0]['text'])
    return response['choices'][0]['text']