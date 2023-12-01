import os
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.environ['API_KEY']
from products_data_english import products_data


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]

def check_moderation(message):
    response = openai.Moderation.create(input=message)
    moderation_output = response["results"][0]
    return moderation_output

def evaluate_response(customer_message, agent_response):
    system_message = """
    You are an assistant that evaluates whether \
    customer service agent responses sufficiently \
    answer customer questions, and also validates that \
    all the facts the assistant cites from the product \
    information are correct.
    The product information and user and customer \
    service agent messages will be delimited by \
    3 backticks, i.e. ```.
    Respond with a Y or N character, with no punctuation:
    Y - if the output sufficiently answers the question \
    AND the response correctly uses product information
    N - otherwise

    Output a single letter only.
    """
    q_a_pair = f"""
    Customer message: ```{customer_message}```
    Product information: ```{products_data}```
    Agent response: ```{agent_response}```

    Does the response use the retrieved information correctly?
    Does the response sufficiently answer the question

    Output Y or N
    """
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': q_a_pair}
    ]

    response = get_completion_from_messages(messages, max_tokens=1)
    return response

# Main method
def main():
    customer_message = """
    tell me about the smartx pro phone and \
    the fotosnap camera, the dslr one. \
    Also tell me about your tvs
    """

    final_response_to_customer = f"""
    The SmartX ProPhone has a 6.1-inch display, 128GB storage, \
    12MP dual camera, and 5G. The FotoSnap DSLR Camera \
    has a 24.2MP sensor, 1080p video, 3-inch LCD, and \
    interchangeable lenses. We have a variety of TVs, including \
    the CineView 4K TV with a 55-inch display, 4K resolution, \
    HDR, and smart TV features. We also have the SoundMax \
    Home Theater system with 5.1 channel, 1000W output, wireless \
    subwoofer, and Bluetooth. Do you have any specific questions \
    about these products or any other products we offer?
    """
    
    another_response = "Life is like a box of chocolates." 

    moderation_output_initial = check_moderation(final_response_to_customer)
    evaluation_result_initial = evaluate_response(customer_message, final_response_to_customer)

    moderation_output_another = check_moderation(another_response)
    evaluation_result_another = evaluate_response(customer_message, another_response)

    print("Moderation Output (Initial Response):", moderation_output_initial)
    print("Evaluation Result (Initial Response):", evaluation_result_initial)
    
    print("Moderation Output (Another Response):", moderation_output_another)
    print("Evaluation Result (Another Response):", evaluation_result_another)

if __name__ == "__main__":
    main()