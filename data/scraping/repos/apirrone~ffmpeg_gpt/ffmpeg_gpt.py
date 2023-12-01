import openai


def get_bash_script(prompt, model="gpt-4"):
    preprompt = """
    You are a bash and ffmpeg expert, you never make mistakes. 
    Write a bash script that uses ffmpeg to perform the following actions.
    Do not explain yourself, just write the script.
    Be sure the script is working. 
    If not specified, the input file is named input.mp4 and the output file is named output.mp4.
    Here are the actions requested : 


    """
    prompt = preprompt + prompt

    res = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content
