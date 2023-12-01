# Initial:
import time
import openai
import os

openai.api_key  = ""
def get_completion(prompt, model="gpt-3.5-turbo"): # Andrew mentioned that the prompt/ completion paradigm is preferable for this class
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def prompts(text, **kwargs):
    summarize_prompt = f"""
    Summarize the text delimited by triple backticks \ 
    into a single sentence using the language of it.
    ```{text}```
    """
    inffering_prompt = f"""
    Identify the following items from the review text:
    - Sentiment (positive or negative)
    - Is the reviewer expressing anger? (true or false)
    - Item purchased by reviewer
    - Company that made the item

    The review is delimited with triple backticks.
    Format your response as a JSON object with "Sentiment", "Anger", "Item" and "Brand" as the keys.
    If the information isn't present, use "unknown" as the value.
    Make your response as short as possible using the language of it.
    Format the Anger value as a boolean.
    The answer is a dictionary containing the following keys with its values: 
    - sentiment: The value is "positive" or "negative"
    - is_anger: The value is "true" or "false"
    - item_purchased: The name of the product reviewing
    - company_name: The name of the company
    Example:
    The content of the text is: "Mình đặt chiếc bánh làm theo mẫu và trang trí thêm hoa quả tại Tiệm bánh Fr*nch P*stry. \
    Đến hôm nhận bánh mang chỉ được như này mà hết tổng cộng 250k. Mình đã gọi cho tiệm hỏi thì tiệm nói với giọng rất là thái độ n..."
    Return:  
    {{
    "sentiment": "negative",
    "is_anger": false,
    "item_purchased": "bánh",
    "company_name": "Fr*nch P*stry"
    }}
    Review text: '''{text}'''
    """
    language_prompt = f"""
    Tell me what language this is and return only the name of the language:
    ```{text}```
    """
    transforming_prompt = f"""
    Proofread and correct the following text and rewrite the corrected version. \
    If you don't find any errors, just return the original version. 
    Translate the text from slang to a business letter using the language is {kwargs.get('language')}.
    Don't use any punctuation around the text:
    ```{text}```
    """
    mail_reply_prompt = f"""
    You are a customer service AI assistant.
    Your task is to send an email reply to a valued customer.
    Given the customer email delimited by ```, \
    Generate a reply to thank the customer for their review.
    If the sentiment is positive or neutral, thank them for their review.
    If the sentiment is negative, apologize and suggest that \
    they can reach out to customer service.
    Make sure to use specific details from the review.
    Write in a concise and professional tone.
    Sign the email as `AI customer agent`.
    Review sentiment: {kwargs.get('sentiment')}
    Customer review: ```{text}```
    """
    return {"summarize":summarize_prompt, "inffering":inffering_prompt, "original_language":language_prompt, "transforming":transforming_prompt, "mail_reply":mail_reply_prompt}
import json
if __name__ == '__main__':
    # Writing prompt
    exercise = str(input("Number of the excercise that you want to check: "))
    text = str(input("The contents: "))
    prompt_dict = prompts(text)
    if "1" in exercise:
        response = get_completion(prompt_dict["summarize"])
        print("summarize response: ", response)
    elif "2" in exercise:
        try:
            prompt_dict = prompts(text)
            prompt = "summarize"
            summarize_response = get_completion(prompt_dict[prompt])
            print(f"{prompt} response: ", summarize_response)
        except:
            time.sleep(60) # Limit of chatgpt accounts allowed
            summarize_response = get_completion(prompt_dict[prompt])

        try:
            prompt_dict = prompts(text)
            prompt = "inffering"
            inffering_response = json.loads(get_completion(prompt_dict[prompt]))
            print(f"{prompt} response: ", inffering_response)
        except:
            time.sleep(60) # Limit of chatgpt accounts allowed
            inffering_response = get_completion(prompt_dict[prompt])

        try:
            prompt_dict = prompts(text)
            prompt = "original_language"
            language_response = get_completion(prompt_dict[prompt])
            print(f"{prompt} response: ", language_response)
        except:
            time.sleep(60) # Limit of chatgpt accounts allowed
            language_response = get_completion(prompt_dict[prompt])

        try:
            prompt_dict = prompts(text, language=language_response)
            prompt = "transforming"
            print(prompt_dict[prompt])
            transforming_response = get_completion(prompt_dict[prompt])
            print(f"{prompt} response: ", transforming_response)
        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(60) # Limit of chatgpt accounts allowed
            transforming_response = get_completion(prompt_dict[prompt])

        try:
            prompt_dict = prompts(text, sentiment=inffering_response["sentiment"])
            prompt = "mail_reply"
            mail_reply_response = get_completion(prompt_dict[prompt])
            print(f"{prompt} response: ", mail_reply_response)
        except:
            time.sleep(60) # Limit of chatgpt accounts allowed
            mail_reply_response = get_completion(prompt_dict[prompt])
