import openai
from better_profanity import profanity

profanity.load_censor_words()


def inspect_transcript(transcript):
    """
    Checks if the given transcript contains any inappropriate content using GPT-3.5.
    Returns False if any inappropriate content is found, otherwise returns True.
    """
    # Check for bad words using the better_profanity library
    contains_bad_word = profanity.contains_profanity(transcript)
    print("Contains bad word?:", contains_bad_word)
    if contains_bad_word:
        return False

    # Create a prompt for GPT-3.5
    prompt = f"Review this text for any inappropriate or innuendo content: '{transcript}'. If nothing inappropriate is found, please respond with only a 1. If anything inappropriate is found, respond with a 0. Do not respond with anything else besides a 0 or 1 even if its very inappropriate. If its too inappropriate for you to even respond, respond with a 0."

    # Use openai.ChatCompletion to get a response from GPT-3.5
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )

    # Extract the GPT-3.5 response
    gpt_response = response["choices"][0]["message"]["content"].strip()
    print("GPT response:", gpt_response)

    # Check the response and return the appropriate value
    if gpt_response == "1":
        print("Passes inspection: True")
        return True
    else:
        print("Passes inspection: False")
        return False


# Example usage:
# result = inspect_transcript("Some innocent text here.")
# print(result)
