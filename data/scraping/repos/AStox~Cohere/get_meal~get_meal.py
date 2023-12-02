import os
import json
import cohere
from weaviate_client import query_weaviate


def lambda_handler(event, context):
    # Load API Key
    api_key = os.getenv("COHERE_API_KEY")

    # Initialize Cohere Client
    co = cohere.Client(api_key)

    # Query Weaviate
    protein_documents = query_weaviate("protein", 3)
    vegetable_documents = query_weaviate("vegetable", 6)
    carb_documents = query_weaviate("carbohydrate", 2)
    documents = protein_documents + vegetable_documents + carb_documents

    chat_history = [
        {
            "role": "USER",
            "message": """Use RAG and the provided documents containing grocery sale information to generate a recipe using as many of the items as reasonably possible.
                You should prioritize making a realistic recipe over using as many items as possible however. 
                Feel free to add in items that aren't on sale if you think it will make the recipe more realistic. 
                And tell me the pricing information for each ingredient where this information can be cited using the attached documents. 
                If you don't know an ingredients price then just say N/A. Here's an example recipe. 
                Always follow an identical format when responding and only respond with a recipe. No extra words.

                ## Sweet Potato and Chicken Hash

                **Ingredients:**
                - 2 sweet potatoes
                - 4 chicken breasts
                - 1 red onion
                - 1 zucchini
                - 1 head of broccoli
                - 1/2 cup of cooked brown rice
                - 1/4 cup of olive oil
                - 1/2 teaspoon of salt
                - 1/4 teaspoon of black pepper

                **Instructions:**
                1. Preheat oven to 425°F.
                2. Chop all vegetables.
                3. In a large bowl, toss sweet potatoes, zucchini, onion, and broccoli with olive oil, salt, and pepper.
                4. Spread the vegetables on a baking sheet and roast in the oven for 25 minutes.
                5. Cook the brown rice as per the instructions on the package.
                6. Meanwhile, heat a large non-stick skillet over medium-high heat and cook the chicken breasts for 6-8 minutes on each side or until cooked through.
                7. Once the vegetables are roasted, add the rice and chicken to the bowl and toss to combine.
                8. Serve immediately and enjoy!

                **Pricing Information:**
                - Sweet Potato (price: $1.12, Savings: $3.27)
                - Chicken Breast (price: $4.61, Savings: $18.52)
                - Red Onion (price: $1.32, Savings: $4.61)
                - Zucchini (price: $1.08, Savings: $4.85)
                - Broccoli (price: N/A)
                - Brown Rice (price: N/A)
                - Olive Oil (price: N/A)
                - Salt (price: N/A)
                - Black Pepper (price: N/A)

                Total Savings: $31.25

                """,
        },
    ]
    message = "Generate the first recipe"
    response = co.chat(
        chat_history=chat_history,
        message=message,
        documents=documents,
        temperature=0.9,
    )

    # Return response
    return {"statusCode": 200, "body": json.dumps(response.text)}
    # print("First Response:")
    # print(response.text)
    # altered_message = "Generate a full 7 day dinner meal plan for me. Start with just the first meal. Use RAG and the provided documents containing grocery sale information to generate a recipe using as many of the items as reasonably possible. You should prioritize making a realistic recipe over using as many items as possible however. Feel free to add in items that aren't on sale if you think it will make the recipe more realistic. And tell me the pricing information for each ingredient where this information can be cited using the attached documents. If you don't know an ingredients price then just say N/A."
    # chat_history.append(
    #     {
    #         "role": "USER",
    #         "message": altered_message,
    #     }
    # )
    # chat_history.append(
    #     {
    #         "role": "CHATBOT",
    #         "message": response.text,
    #     }
    # )

    # message = """Now generate the next meal. Base it around a different protein than the other recipes but follow the exact same format as the other recipes. Make sure to include price information for each ingredient where possible. If you don't know the price of an ingredient then just say N/A."""
    # response = co.chat(
    #     chat_history=chat_history,
    #     message=message,
    #     documents=documents,
    #     temperature=0.9,
    # )
    # print("--------------------------------------")
    # print("\n\n\n Second Response:")
    # print(response.text)

    # chat_history.append(
    #     {
    #         "role": "USER",
    #         "message": message,
    #     }
    # )
    # chat_history.append(
    #     {
    #         "role": "CHATBOT",
    #         "message": response.text,
    #     }
    # )

    # message = """Now generate the next meal. Base it around a different protein than the other recipes but follow the exact same format as the other recipes. Make sure to include price information for each ingredient where possible. If you don't know the price of an ingredient then just say N/A."""
    # response = co.chat(
    #     chat_history=chat_history,
    #     message=message,
    #     documents=documents,
    #     temperature=0.9,
    # )
    # print("--------------------------------------")
    # print("\n\n\n Third Response:")
    # print(response.text)
