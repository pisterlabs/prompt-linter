import openai

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
openai.api_key = 'ADD_OPENAI_KEY'
openai.api_base = 'http://localhost:8081/v1/openai'


def call_openai_api(prompt):
    try:
        response = openai.Completion.create(
            # You can use other engines, but 'text-davinci-002' is powerful (as of my knowledge cutoff in September 2021).
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=150,
            headers={
                "X-Numexa-Api-Key": "<ADD_NUMEXA_KEY>",
            }
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        return str(e)


def main():
    while True:
        prompt = input("Enter your prompt (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break

        response = call_openai_api(prompt)
        print("OpenAI API Response:")
        print(response)
        print("\n")


if __name__ == "__main__":
    main()

# def call_openai_api(prompt):
#     try:
#         response = openai.Completion.create(
#             stream=True,
#             engine="text-davinci-002",
#             prompt=prompt,
#             max_tokens=150,
#             headers={
#                 "X-Numexa-Api-Key": "nxa-QMwEXoetrVNV0GRICsUlxdmCMpbi1bkm",
#             }
#         )

#         # Iterate through the generator object to get partial responses
#         for partial_response in response:
#             print("Partial OpenAI API Response:")
#             print(partial_response['choices'][0]['text'])
        
#         # Get the final response after the generator is exhausted
#         final_response = response.choices[-1]['text'].strip()
#         return final_response

#     except Exception as e:
#         return str(e)

# def main():
#     while True:
#         prompt = input("Enter your prompt (or 'exit' to quit): ")
#         if prompt.lower() == 'exit':
#             break

#         response = call_openai_api(prompt)
#         print("Final OpenAI API Response:")
#         print(response)
#         print("\n")


# if __name__ == "__main__":
#     main()
