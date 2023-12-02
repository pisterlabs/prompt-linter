import openai
from .ChatModelCompletor import get_completion_from_messages as get_completion_from_messages

class Customer:
    def __init__(self, api_key, products, language="en"):
        self.api_key = api_key
        self.products = products
        self.language = language
        openai.api_key = self.api_key

    def generate_question_or_comment(self):
        # Create a list of product names
        product_list = "##".join(self.products.keys())

        delimiter = "####"

        system_message = f"""
        You are a customer of a store.\
        The store has product list as {product_list} delimited with ## characters: \
        Randomly select fewer than 2 products from the list to inquire. \
        Products should have equal probability of being consulted. \
        You need to query details about the products. \
        The specific requirements will be delimited with \
        {delimiter} characters.\
        Only generate a comment to inquire, with nothing else.\
        """

        user_message_1 = f"""
            Inquire in {self.language} language. Using 100 words at most. \
        """

        messages =  [  
            {'role':'system', 
            'content': system_message},    
            {'role':'user', 
            'content': f"{delimiter}{user_message_1}{delimiter}"},  
        ]

        return get_completion_from_messages(messages)

