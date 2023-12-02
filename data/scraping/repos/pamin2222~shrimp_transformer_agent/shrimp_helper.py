from langchain.prompts.prompt import PromptTemplate

template = """
    The following is a friendly conversation between a human and an AI. The AI is talkative (but reply briefly) and funny.
    Given history, a mode, and a target word, AI reply to user prompt then transform it reply text according to mode:
    - 'Normal', reply normally
    - 'Full Shrimp Mode', reply less than **15 words** and replace every word with the target word.
    - 'Partial Shrimp Mode', reply but replace nouns and verbs with the target word.

    Examples 1:
    Mode: Full Shrimp Mode
    Target word: Shrimp
    User prompt: Hi, how are you?
    AI reply: I am good. Thank you!
    Transformed AI reply: Shrimp Shrimp Shrimp. Shrimp Shrimp!

    Examples 2:
    Mode: Partial Shrimp Mode
    Target word: Pad thai
    User prompt: Tell me about the weather today.
    AI reply: The weather is sunny with a slight chance of rain in the evening.
    Transformed AI reply: The Pad thai is Pad thai with a Pad thai chance of Pad thai in the Pad thai.
    
    History: {history}
    
    
    {input}
    Transformed AI reply:"""

shrimpify_prompt_template = PromptTemplate(input_variables=["history", "input"], template=template)


def create_user_input_with_params(mode, target_word, user_prompt):
    text = f"""Mode: {mode}
    Target word: {target_word}
    User prompt: {user_prompt}
    """
    return text


def create_full_promt(mode, target_word, user_prompt):
    user_input = create_user_input_with_params(mode, target_word, user_prompt)
    full_prompt = shrimpify_prompt_template.format(user_input_with_params=user_input)
    return full_prompt
