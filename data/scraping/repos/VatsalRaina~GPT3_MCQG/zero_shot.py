import argparse
import os
import sys
import openai

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--save_path', type=str, help='Load path to which trained model will be saved')
parser.add_argument('--context_path', type=str, help='Load path to contexts.txt')
parser.add_argument('--part', type=int, default=1, help='Specify the segment of 100')
parser.add_argument('--openai_access_key', type=str, help='Access key from OpenAI')

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    openai.api_key = args.openai_access_key

    with open(args.context_path, 'r') as f:
        all_contexts = [a.rstrip() for a in f.readlines()]

    all_responses = []

    count = 0

    # Do 100 at a time for safety
    val = 100
    start = (args.part-1)*val
    end = min(args.part * val, len(all_contexts))

    for context in all_contexts[start:end]:
        response = openai.Completion.create(
        model="text-davinci-002",
        prompt="Multiple-choice question with 4 options and an answer.\n\n"+context,
        temperature=0.4,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )

        response_text = response["choices"][0]["text"].replace("\n", " [SEP] ")

        all_responses.append(response_text)

        count += 1
        print(count)
        # if count == 3:
        #     break

    with open(args.save_path+str(args.part)+"_responses.txt", 'w') as f:
        f.writelines("%s\n" % resp for resp in all_responses)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


