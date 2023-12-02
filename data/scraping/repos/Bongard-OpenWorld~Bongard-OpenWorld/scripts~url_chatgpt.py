import os
import re
import json
import copy
import random
import openai

# Load your API key from an environment variable or secret management service
api_key = '<your OpenAI API key>'
openai.api_key = api_key

prompt_base = '''Given 6 "positive" images url and 6 "negative" images url, please complete the following task based on the content of images. Where "positive" images can be summarized as 1 "common" sentence and "negative" images cannot, the "common" sentence describes a set of concepts that are common to "positive" images. And then given 1 "query" image url, please determine whether the "query" belongs to "positive" or "negative" and give the "common" sentence from "positive" images.

Please complete the following query:
'''

url_path = 'assets/data/bongard-ow/bongard_test.json'
save_path = 'url_chatgpt.json'

def main():
    query_list = []

    with open(url_path, 'r') as f:
        bongard_ow = json.load(f)
        for sample in bongard_ow:
            uid = sample['uid']
            commonSense = sample['commonSense']
            concept = sample['concept']
            caption = sample['caption']
            imageFiles = sample['imageFiles']

            urls = []
            for image in imageFiles:
                url_list = image.split('/')[-1].split('__')[3:]
                url = f"{url_list[0]}//{'/'.join(url_list[1:])}"
                urls.append(url)

            positive = urls[:6]
            query_a = urls[6]
            negative = urls[7:13]
            query_b = urls[13]

            query = {}
            query['commonSense'] = commonSense
            query['concept'] = concept
            query['caption'] = caption
            query['positive'] = positive
            query['negative'] = negative
            
            query['uid'] = uid + '_A'
            query['query'] = query_a
            query_list.append(copy.deepcopy(query))

            query['uid'] = uid + '_B'
            query['query'] = query_b
            query_list.append(copy.deepcopy(query))

        random.shuffle(query_list)

        summary = []
        for query in query_list:
            prompt = copy.deepcopy(prompt_base)
            prompt += 'positive: ' + str(query['positive']) + '\n'
            prompt += 'negative: ' + str(query['negative']) + '\n'
            prompt += 'query: ' + str(query['query']) + '\n\n'

            prompt += 'Answer:\npositive or negative:'

            try:
                response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                        messages=[{
                                                            "role": "user",
                                                            "content": prompt
                                                        }],
                                                        max_tokens=1024,
                                                        temperature=1,
                                                        n=1,
                                                        frequency_penalty=0,
                                                        presence_penalty=0)

                print(response, '\n')
                text = response['choices'][0]['message']['content'] + '\n'

                answer = re.findall('(.*?)\n', text)
                sentence = re.findall(':(.*?)\n', text)

                query['answer'] = answer[0].lower()
                query['sentence'] = sentence[-1].strip()

                summary.append(copy.deepcopy(query))
            except Exception as e:
                    print(f'response eorror: {e}')

        with open(save_path, "w") as file:
            json.dump(summary, file, indent=4)

if __name__ == '__main__':
    main()