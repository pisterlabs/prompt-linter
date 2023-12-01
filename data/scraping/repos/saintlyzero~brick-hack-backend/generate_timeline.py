import cohere
from data import OOP, OS_CN
from keys import API_KEY
import cohere


def generate_timeline(co):
    response = co.generate(
        model='command-xlarge-20221108',
        prompt=f'extract all concepts from lecture: {OS_CN}',
        max_tokens=200,
        temperature=0,
        k=0,
        p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=[],
        return_likelihoods='NONE')
    return response.generations[0].text


def main():
    co = cohere.Client(API_KEY)
    print(generate_timeline(co))


if __name__ == '__main__':
    main()
