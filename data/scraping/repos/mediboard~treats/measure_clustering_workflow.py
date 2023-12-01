import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import openai
import os
import itertools
from tqdm import tqdm
import openai

Secret = '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'
openai.organization = "org-j6fGVx3OgjgpAbCQFHOmdEUe"
openai.api_key = Secret

prompt = '''Write a title that describes this group of titles in the form "title: <title>"
group: [
{}
]

'''

def generate_responses(prompts):
  # Initialize list to store responses
  # Process prompts in batches
  completions = openai.Completion.create(
    engine="text-davinci-003",
    prompt=[x for x in prompts],
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.1)

  # Extract responses from OpenAI API response
  return [c.text.strip() for c in completions.choices]


def chunks(iterable, batch_size=100):
  """A helper function to break an iterable into chunks of size batch_size."""
  it = iter(iterable)
  chunk = tuple(itertools.islice(it, batch_size))
  while chunk:
    yield chunk
    chunk = tuple(itertools.islice(it, batch_size))


def create_cluster_names(embedded_measures):
  embedded_measures['label_count'] = embedded_measures['label'].map(embedded_measures['label'].value_counts())

  measure_labels = {
    'label': [],
    'prompt': []
  }

  for label in embedded_measures[embedded_measures['label_count'] > 10]['label'].drop_duplicates():
    measure_labels['label'].append(label)
    measure_labels['prompt'].append(prompt.format('\n'.join(embedded_measures[embedded_measures['label'] == label]['title'].sample(10))))

  measure_labels = pd.DataFrame.from_dict(measure_labels)

  titles = []
  for prompts in tqdm([list(c) for c in chunks(measure_labels['prompt'], batch_size=20)]):

    titles += generate_responses(prompts)

  measure_labels['cluster_title'] = pd.Series(titles)
  measure_labels['cluster_title'] = measure_labels['cluster_title'].str.replace("Title: ", "")

  return measure_labels


def load_embeded_measures():
  return pd.read_pickle("encoded_titles.pkl")


def cluster_embeddings(embedded_measures):
  X = np.vstack(embedded_measures['vector'])

  clustering = AgglomerativeClustering(linkage='average', distance_threshold=.5, n_clusters=None)

  # fit the clustering model to the data
  labels = clustering.fit_predict(X)

  embedded_measures['label'] = labels

  return embedded_measures


def load_measures(connection):
  measures = pd.read_sql("select * from measures", connection)

  return measures


def load_measure_groups(connection):
  measure_groups = pd.read_sql("select * from measure_groups", connection)

  return measure_groups


def upload_to_db(data: pd.DataFrame, table_name, connection):
  data.to_sql(table_name, connection, index=False, if_exists='append', schema='public')   


def create_measure_groups(measure_titles):
  # Create the measure groups
  # Create the measure group measures

  measure_groups = measure_titles[['cluster_title']].drop_duplicates()
  measure_groups['id'] = range(1, len(measure_groups) + 1)

  measure_groups = measure_groups.rename(columns={'cluster_title': 'name'})

  return measure_groups


def create_measure_group_measures(measure_titles, connection):
  measure_groups = load_measure_groups(connection)

  merged = measure_titles.merge(
    measure_groups,
    left_on=['cluster_title'],
    right_on=['name'],
    suffixes=['_measure', '_group'])

  print(merged)

  measure_group_measures = merged[['id_measure', 'id_group']].drop_duplicates()

  measure_group_measures['id'] = range(1, len(measure_group_measures) + 1)
  measure_group_measures = measure_group_measures.rename(columns={
    'id_measure': 'measure',
    'id_group': 'measureGroup'
  })

  return measure_group_measures


def measure_clustering_workflow(connection):
  embedded_measures = load_embeded_measures()

  clustered_measures = cluster_embeddings(embedded_measures)

  measure_labels = create_cluster_names(clustered_measures)

  titled_measures = clustered_measures.merge(measure_labels, on='label')

  measures = load_measures(connection)

  measure_title_measures = measures.merge(titled_measures, on=['title'])

  measure_groups = create_measure_groups(measure_title_measures)

  upload_to_db(measure_groups, "measure_groups", connection)

  measure_group_measures = create_measure_group_measures(measure_title_measures, connection)

  upload_to_db(measure_group_measures, "measure_group_measures", connection)


