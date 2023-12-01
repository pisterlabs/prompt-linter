import random
import openai

# Set up the OpenAI API credentials
openai.api_key = ""


# Function to generate advertisement prompts
def generate_advertisement(product_service_name, keywords, length="medium", organization_name="0"):
    """
    Generates advertisement prompts for a given product or service.

    Args:
        product_service_name (str): The name of the product or service.
        keywords (str): Keywords to include in the advertisement prompts.
        length (str, optional): The desired length of the prompts. Can be "short", "medium", or "long". Defaults to "medium".
        organization_name (str, optional): The name of the organization. Defaults to "0".

    Returns:
        tuple: A tuple containing a list of generated advertisement prompts and a list of corresponding tones.
    """
    tones = ["persuasive", "exciting", "funny"]

    # List of tones appropriate for advertisements
    prompts = [
        f"Generate an advertisement prompt for {product_service_name}. Make it persuasive for the reader to get the want of immediately purchasing it. Keywords to include are {keywords}. The length must be {length}.",
        f"Generate an advertisement prompt for {product_service_name}. Make it exciting for the reader to get excited about the {product_service_name} and get the want of immediately purchasing it. Keywords to include are {keywords}. The length must be {length}.",
        f"Generate an advertisement prompt for {product_service_name}. Make it slightly funny for the reader to get in a good mood and positively remember the {product_service_name}. Keywords to include are {keywords}. The length must be {length}.",
    ]

    model = "text-davinci-002"
    temperature = 0.5

    if length == "short":
        max_tokens = 240
    elif length == "medium":
        max_tokens = 480
    elif length == "long":
        max_tokens = 720

    indices = list(range(len(tones)))
    random.shuffle(indices)

    tones = [tones[i] for i in indices]
    prompts = [prompts[i] for i in indices]

    ad_prompts = []
    for i in range(len(tones)):
        prompt = prompts[i]

        if organization_name != "0":
            prompt += f" This advertisement is for {organization_name}."
        
        # Call the OpenAI API to generate the advertisement prompts
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            frequency_penalty=0,
            presence_penalty=0,
        )

        # Extract the generated prompts from the response
        prompt_choices = [choice.text.strip() for choice in response.choices]
        ad_prompts.append(prompt_choices[0])

    return ad_prompts, tones
