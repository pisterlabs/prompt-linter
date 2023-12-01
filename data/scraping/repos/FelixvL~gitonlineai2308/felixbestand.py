import openai



def ditismijnfunctie(key, stad):
    openai.api_key = key


    prompt = stad


    response = openai.Completion.create(
    engine='text-davinci-003',  # Specify the language model you want to use
    prompt=" geef de vijf top siteseeings van de stad:  "+prompt,
    max_tokens=500  # Adjust the desired length of the completion
    )


    generated_text = response.choices[0].text.strip()

    return "methode van felix: "+ generated_text