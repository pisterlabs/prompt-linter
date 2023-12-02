import os
import re
import json
import copy
import random
import openai

# Load your API key from an environment variable or secret management service
api_key = '<your OpenAI API key>'
openai.api_key = api_key

prompt_base = '''Given 6 "positive" sentences and 6 "negative" sentences, where "positive" sentences can be summarized as 1 "common" sentence and "negative" sentences cannot, the "common" sentence describes a set of concepts that are common to "positive" sentences. And then given 1 "query" sentence, please determine whether the "query" belongs to "positive" or "negative" and give the "common" sentence from "positive" sentences.

Please complete the following query:
'''

caption_path = 'blip2.json'
save_path = 'blip2_gpt4.json'

def main():
    query_list = []

    with open(caption_path, 'r') as f:
        bongard_ow = json.load(f)
        for sample in bongard_ow:
            uid = sample['uid']
            commonSense = sample['commonSense']
            concept = sample['concept']
            caption = sample['caption']
            imageFiles = sample['imageFiles']
            
            captions = data['captions']
            positive = captions[:6]
            query_a = captions[6]
            negative = captions[7:13]
            query_b = captions[13]

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
                response = openai.ChatCompletion.create(model="gpt-4",
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