import requests
import json
import openai

def get_food_response(image_path):
    url = "https://api.logmeal.es/v2/image/segmentation/complete"

    payload = {}
    files = [
        ('image', ('image.jpg', open(image_path, 'rb'), 'image/jpeg'))
    ]
    headers = {
        'Authorization': 'Bearer eda87bc113be66c5304bfb00a17e2f9862b3f02c'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    # Check if the API response was successful
    if response.status_code != 200:
        print(f"Error: API request failed with status code {response.status_code}")
        print(response.text)
        exit()

    result = response.json()

    # Check if the expected keys are present in the API response
    if 'segmentation_results' not in result:
        print("Error: segmentation_results not found in API response")
        print(result)
        exit()

    if len(result['segmentation_results']) == 0:
        print("Error: No segmentation results found in API response")
        print(result)
        exit()

    if 'recognition_results' not in result['segmentation_results'][0]:
        print("Error: recognition_results not found in segmentation_results")
        print(result)
        exit()

    if len(result['segmentation_results'][0]['recognition_results']) == 0:
        print("Error: No recognition results found in segmentation_results")
        print(result)
        exit()

    dish = result['segmentation_results'][0]['recognition_results'][0]['name']
    print(dish)
    print(result['segmentation_results'][0]['recognition_results'][0]['prob'])

    openai.api_key = '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'
    messages = [{"role": "system", "content": "You are an intelligent assistant."}]

    message = f'Please write a recipe for {dish}'
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
    reply = chat.choices[0].message.content
    output = []
    if reply:
        d = reply.split('.')
        details = d[0]
        ingredients = details.split(':')
        # print(ingredients)
        head1 = ingredients[2]
        dish_name = ingredients[0] + ': ' 
        ingredients = ingredients[2]
        ingredients_terms = ingredients.split('-')
        output.append(dish_name)
        # output.append(head1)
        for i in ingredients_terms:
            output.append(i)
        # output.append(ingredients_terms)
        # output.append(head1)
        instructions = d[1].split(':')

        output.append(instructions[0])
        # output.append(instructions[1])




        # print(dish_name)
        # print('\n')
        # print(head1)
        # for i in ingredients_terms :
        #     print(i)
        # print('\n')
        # print(instructions[0] + ':', end='\n')
        # print(instructions[1] + '.', end = '\n')
        for i in range(2, len(d)):
            output.append(d[i])

    return output

    # Optionally, you can return dish and reply as a tuple if you need both values:
    # return dish, reply
