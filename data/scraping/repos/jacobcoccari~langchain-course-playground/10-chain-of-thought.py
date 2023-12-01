from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def chat(message, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message["content"]


delimiter = "####"
products = """
1. Product: QuantumBook Pro
   Brand: QuantumBook
   Features: 14-inch display, 16GB RAM, 512GB SSD, Intel Core i7 processor
   Description: A stylish and powerful ultrabook for seamless multitasking.
   Price: $899.99

2. Product: ThunderX Gaming Laptop
   Brand: ThunderX
   Features: 17.3-inch display, 32GB RAM, 1TB SSD, NVIDIA GeForce RTX 3080
   Description: An ultimate gaming laptop for unparalleled gaming performance.
   Price: $1499.99

3. Product: FlexiTech Convertible
   Brand: FlexiTech
   Features: 13.3-inch touchscreen, 16GB RAM, 256GB SSD, 360-degree hinge
   Description: A flexible convertible laptop for creativity and productivity.
   Price: $799.99

4. Product: QuantumDesk Pro
   Brand: QuantumDesk
   Features: Intel Core i9 processor, 32GB RAM, 2TB SSD, NVIDIA GeForce RTX 3080
   Description: A high-performance desktop computer for intensive tasks and gaming.
   Price: $1499.99

5. Product: ThunderBook Chromebook
   Brand: ThunderBook
   Features: 12-inch display, 8GB RAM, 128GB eMMC, Chrome OS
   Description: A portable and efficient Chromebook for on-the-go computing.
   Price: $349.99

"""

prompt = f"""

Follow these steps to answer a customer query about a specific product. The query will be delimited using {delimiter}.

Step 1:{delimiter} Identify whether the product the customer is asking about is in the following list:
{products}

Step 2:{delimiter} If the user's message contains a product in the list above, list any assumptions that the user is making. For example, that Laptop Z has a 13 inch screen, or that Laptop X is bigger than Laptop Y.

Step 3:{delimiter}: If the user made any assumptions, figure out whether the assumption is true based on your product information. 

Step 4:{delimiter}: First, politely correct the customer's incorrect assumptions if they made an incorrect assumption. Answer the customer in a friendly tone.

Use the following format:
Step 1:{delimiter} <step 1 reasoning>
Step 2:{delimiter} <step 2 reasoning>
Step 3:{delimiter} <step 3 reasoning>
Response to user:{delimiter} <response to customer>

Make sure to include {delimiter} to separate every step.

User query:

```I am looking for a thunderbook with a 16 inch screen.```
"""

response = chat(prompt)
# print(response)

# Using the delimiter, we can now split the model's response to hide the reasoning.

try:
    final_response = response.split(delimiter)[-1].strip()
    print(final_response)
except Exception as e:
    final_response = (
        "Sorry, I'm having trouble right now, please try asking another question."
    )
