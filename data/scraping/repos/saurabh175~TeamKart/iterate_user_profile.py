import openai 
openai.api_key = "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"

def strengthen_profile(cur_profile, recent_chat, products_bought):

    profile_current = "Here is the current profile (it may be empty): " + cur_profile + "\n"
  
    prompt = f"""
    We are an e-commerce platform that sells products to customers. We want to strengthen our customer profiles by adding information about the products that they have bought and conversations they have had.
    For example, if a customer has bought a product, we want to add information about that product to their profile. If a customer has had a conversation with a customer service representative, we want to add information about that conversation to their profile.
    By adding information, we don't want to just copy the direct product names into their profile - rather, we want to derive insights about the persona and background of the user.
    For example, if the user is buying hiking clothes and talking about playing sports, we can assume that this user is an active individual.
    If the user is buying a lot of books and talking about reading, we can assume that this user is an avid reader.
    If the user talks about keeping warm, the user may live in a cold area, so save that he likes to be warn and might live in a cool environment.
    
    {profile_current}

    Here is their most recent chat - this may be structured like a transcripts.
    {recent_chat}

    Here are the products that the user has bought:
    {products_bought}

    Format your answer in the following way:
    User is a ______

    New user profile: 
    """

    
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt= prompt,
        temperature=0.3,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response['choices'][0]




