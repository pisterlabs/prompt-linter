import pickle
from pathlib import Path
from time import sleep

import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from natsort import natsorted
from scipy.spatial.distance import cosine
from tqdm import tqdm

from jane_watson.util import extract_text_from_subtitle

load_dotenv()


def summarize_in_bullets(text, retries=5, sleep_between_retries=1):
    for i in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": f"Class transcript:\n{text}\n\n\n"
                               f"Summarize this class in bullet points."
                }],
                temperature=0,
                max_tokens=256,
            )
            return response.choices[0].message.content
        except Exception as e:
            sleep(sleep_between_retries)
            print(f'Retrying {i + 1} of {retries}, {str(e)}')
    raise Exception(f'Failed to summarize {text}')


def get_chat_response(text, retries=5, sleep_between_retries=1):
    for i in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": text
                }],
                temperature=0,
                max_tokens=256,
            )
            return response.choices[0].message.content
        except Exception as e:
            sleep(sleep_between_retries)
            print(f'Retrying {i + 1} of {retries}, {str(e)}')
    raise Exception(f'Failed to summarize {text}')


def get_embeddings(text, retries=5, sleep_between_retries=1):
    for i in range(retries):
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response['data'][0]['embedding']
        except openai.error.ServiceUnavailableError as e:
            sleep(sleep_between_retries)
            print(f'Retrying {i + 1} of {retries}, {str(e)}')
    raise Exception(f'Failed to summarize {text}')


def extract_module(
        module_path,
        first_class_id,
        module_id,
        course_id=39458,
        show_progress=False
):
    module_path = Path(module_path)
    module_name = module_path.stem.split('___')[1]\
        .replace('_', ' ').replace(' subtitles', '')

    files = natsorted(list(module_path.glob('*.srt')))
    pbar = tqdm(files) if show_progress else files
    data = []
    for i, str_file in enumerate(pbar):
        class_name = str_file.stem.split(' - ')[1]
        if show_progress:
            pbar.set_postfix(class_name=class_name, module_name=module_name)
        if class_name.endswith('Question'):
            continue
        if class_name.endswith('Solution'):
            solution = extract_text_from_subtitle(str_file)
            question = extract_text_from_subtitle(str(files[i - 1]))
            text = f'{question}\n{solution}'
        else:
            text = extract_text_from_subtitle(str_file)

        summary = summarize_in_bullets(text)
        text_for_embeddings = f'Class title: {class_name}\n\n' \
                              f'Module: {module_name}\n\n' \
                              f'Class summary:\n{summary}\n\n' \
                              f'Class transcript:\n{text}'
        embeddings = get_embeddings(text_for_embeddings)
        data.append({
            'module': module_name,
            'class': class_name,
            'summary': summary,
            'transcript': text,
            'embeddings': embeddings
        })
    return data


def extract_course(
        course_path,
        first_class_id=348259,
        first_module_id=62750,
        course_id=39458
):
    course_path = Path(course_path)
    modules_paths = natsorted(list(course_path.glob('*___*_subtitles')))
    data = []
    fcid = first_class_id
    for i, module_path in enumerate(modules_paths):
        module_name = module_path.stem.split('___')[1]\
            .replace('_', ' ').replace(' subtitles', '')
        print(f'Extracting module {i + 1}: {module_name}...')
        module_id = first_module_id + i
        module_data = extract_module(
            module_path,
            first_class_id=fcid,
            module_id=module_id,
            course_id=course_id
        )
        data += module_data
        fcid += len(module_data) + 1
        with Path(f'/tmp/debug_kbai_{i + 1}.pkl').open('wb') as f:
            pickle.dump(data, f)

    return data


def answer(query: str, data: pd.DataFrame, n_top: int = 10):
    query_embeddings = np.array(get_embeddings(query))
    data['embeddings'] = data['embeddings'].apply(np.array)
    data['distance'] = data['embeddings'].apply(lambda x: cosine(x, query_embeddings))
    data['similarity'] = 1 - data['distance']
    data['similarity'] = data['similarity'].apply(lambda x: f'{x:.1%}')
    results = data.sort_values('distance')
    results['class'] = results['class'].str.replace('_', ':')

    top = results.iloc[0]
    question = f'Class title: {top["class"]}\n\n' \
               f'Module: {top["module"]}\n\n' \
               f'Class summary:\n{top["summary"]}\n\n' \
               f'Class transcript:\n{top["transcript"]}\n\n\n' \
               f'Based on the information above, answer the following question:\n' \
               f'{query}\n\n' \
               f'Inform the module name and the class name in the answer.'

    columns = ['module', 'class', 'summary', 'similarity']
    return {
        'answer': get_chat_response(question),
        'top_results': results.head(n_top)[columns].to_dict('records'),
    }


if __name__ == '__main__':
    path = '/Volumes/Mac/OMSCS'
    d = extract_course(path)
    with Path('kbai.pkl').open('wb') as f:
        pickle.dump(d, f)
