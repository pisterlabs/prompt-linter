import os
import openai
import tiktoken

def question_generate(text):
    """
    Generate question and answer pair from given text.
    
    Parameters
    ----------
    text : str
        The text you want to generate question and answer pair.
        
    Returns
    -------
    str
        The question and answer pair generated from given text.
    """
    prompt = """<|im_sep|>
    System: You are a AI assistant help people generate question and answer pair from given text,
    the question and answer will be used to train a chatbot,
    response must be json in format: [{{""question"": ""<question goes here>"", ""answer"": ""<answer goes here>""}}].
    User: I want to generate question and answer pair from given text.
    System: Please input the text you want to generate question and answer pair.
    User: {}
    System: Please wait a moment, I am generating question and answer pair from given text.
    <|im_end|>
    """.format(text)
    
    openai.api_type = "azure"
    openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_version = "2022-12-01"
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(text))
    print(num_tokens)

    response = openai.Completion.create(
        engine=deployment_name,
        prompt=prompt,
        temperature=0.3,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["<|im_end|>"]
    )

    return response.choices[0].text
