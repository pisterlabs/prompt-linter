import json
import openai
from openai.embeddings_utils import get_embedding
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from preprocessor import bugs_preprocessor
import os
from collections import OrderedDict
from api_key import key as your_key_here


class ADAEncoder(TransformerMixin, BaseEstimator):

    def __init__(self, target_path):
        self.engine = "text-embedding-ada-002"
        openai.api_key = your_key_here
        self.target_path = target_path

    def transform(self, X, y):
        for i, x in enumerate(X):
            print('Working on {}'.format(y[i]))
            emb = get_embedding(x, engine=self.engine)
            # emb = []
            data = {'pr_num': y[i], 'embedding': emb}

            with open(self.target_path + '/{}_ada_embedding.json'.format(y[i]), 'w') as f:
                json.dump(data, f)

            print('Saved: {}'.format(y[i]))


def extract_columns(properties):
    """
    Returns list made of synopsis (str), description (str), component (dict)
    :param properties:
    :return: list
    """
    description = properties.get('pr_description', '')
    if not description:
        description = ''

    if 'data:image/png;base64' in description:
        description = description.split('data:image/png;base64')
        description = description[0]

    description = description.split(' ')
    description = list(filter(None, description))

    if len(description) > 256:
        description = ' '.join(description[:256])
    else:
        description = ' '.join(description)

    text = properties.get('synopsis') + ' ' + description + ' ' + properties.get('category', '')
    tmp = bugs_preprocessor(text)
    if len(tmp) > 8191:
        tmp = tmp[:8090]
    text = ' '.join(tmp)

    return text


def get_X_y(pr_data):
    X = []
    y = []
    print('Extracting..')
    pbar = tqdm(total=len(pr_data))
    for pr, properties in pr_data.items():
        text = extract_columns(properties)
        X.append(text)
        y.append(pr)
        pbar.update(1)
    pbar.close()
    return X, y


def load_pr_data(data_path):
    """
    Creates json object from description and parameters files.
    :param data_path
    :return: dictionary
    """
    pr_data = OrderedDict()
    files_to_load = os.listdir(data_path)

    if '.DS_Store' in files_to_load:
        files_to_load.remove('.DS_Store')

    files_to_load = sorted(files_to_load, key=lambda i: int(i), reverse=True)
    for i, pr_number in enumerate(files_to_load):
        description_file = data_path + '/' + pr_number + '/' + pr_number + '_description.txt'
        properties_file = data_path + '/' + pr_number + '/' + pr_number + '_properties.json'
        if os.path.exists(description_file):
            with open(description_file) as des_file_obj:
                description = des_file_obj.read()

        if os.path.exists(properties_file):
            with open(properties_file) as prop_file_obj:
                pr_features = json.load(prop_file_obj)
                pr_features['product'] = pr_features.get('product', '').strip('\n')
                pr_features['pr_description'] = description
                pr_features.pop('submitter-id')
                pr_data[pr_features.pop('number')] = pr_features

    return pr_data


def main(data_path, target_path):
    pr_data = load_pr_data(data_path)

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # remove any already existing embeddings
    existing_embeddings = [x.replace('_ada_embedding.json', '') for x in os.listdir(target_path)]
    for pr in existing_embeddings:
        del (pr_data[pr])

    print('Embeddings to create: ', len(pr_data))

    X, y = get_X_y(pr_data)
    if not os.path.exists(tpath):
        os.makedirs(tpath)
    encoder = ADAEncoder(target_path=target_path)
    encoder.transform(X, y)


if __name__ == '__main__':
    repo = 'MozillaCore'
    type = 'training'
    spath = '/Users/patila/Desktop/open_data/bugrepo/{}/{}'.format(repo, type)
    tpath = 'ada_embeddings/{}_{}'.format(repo, type)
    main(spath, tpath)
    print('Done')
