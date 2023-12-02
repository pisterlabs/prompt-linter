import openai
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

API_KEY = # ADD API KEY HERE
if not API_KEY:
    print('\n\nERROR: Missing API key\n\n')
openai.api_key = API_KEY


def prompt_gpt(prompt, max_tokens=250, temperature=0.7, model='text-davinci-003'):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1
    )
    return response.choices[0].text


def get_letter_res(res, print_extra=False):
    if 'A)' in res.upper() or 'A:' in res.upper() or 'A.' in res.upper():
        return 0
    if 'B)' in res.upper() or 'B:' in res.upper() or 'B.' in res.upper():
        return 1
    if 'C)' in res.upper() or 'C:' in res.upper() or 'C.' in res.upper():
        return 2
    if 'D)' in res.upper() or 'D:' in res.upper() or 'D.' in res.upper():
        return 3
    if 'E)' in res.upper() or 'E:' in res.upper() or 'E.' in res.upper():
        return 4
    if 'F)' in res.upper() or 'F:' in res.upper() or 'F.' in res.upper():
        return 5
    if print_extra:
        print("Can't find letter: ", res)
    return None


def run_survey(prompt,
               choices=[],
               categories=[],
               iterations=10,
               colors=['g', 'b', 'r'],
               x_label='',
               model='text-davinci-003',
               temperature=0.7,
               print_extra=False,
               max_tokens=256,
               print_res=False):
    num_choices = len(choices)
    results = {}
    for category in categories:
        print('Category: ', category)
        prompt_f = prompt.replace('{category}', category)
        res_array = []
        for _ in tqdm(range(iterations)):
            res = prompt_gpt(prompt_f, max_tokens=max_tokens, model=model, temperature=temperature)
            res_array.append(get_letter_res(res, print_extra=print_extra))
            if print_res:
                print('Response: ', res)
        results[category] = res_array

    # turn results into %
    responses_array = [[] for _ in range(num_choices)]
    for responses in results.values():
        for i in range(num_choices):
            responses_array[i].append(responses.count(i) / len(responses))

    # plot
    bar_width = 0.25
    fig = plt.subplots(figsize=(12, 8))

    bars = [np.arange(len(responses_array[0]))]
    for i in range(1, num_choices):
        bars.append([x + bar_width for x in bars[i - 1]])

    for i in range(len(bars)):
        plt.bar(bars[i], responses_array[i], color=colors[i % len(colors)], width=bar_width, edgecolor='grey',
                label=choices[i])

    plt.xlabel(x_label, fontweight='bold', fontsize=15)
    plt.ylabel('%', fontweight='bold', fontsize=15)
    plt.xticks([r + bar_width for r in range(len(responses_array[0]))], categories)

    plt.legend()
    plt.show()
    plt.savefig('output/new_graph.pdf', format='pdf')

    res_list = [item if item is not None and item < len(categories) else None for sublist in results.values() for item
                in sublist]
    # print('res_list: ', res_list)
    print(f"""
  ITERATIONS: {iterations}
  TEMPERATURE: {temperature}
  Unrecognized Responses: {round(100 * (res_list.count(None) / len(res_list)), 2)}%
  """)
    #   results: {results}
