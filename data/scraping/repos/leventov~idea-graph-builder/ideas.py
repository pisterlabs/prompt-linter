from typing import List, Tuple
from dataclasses import dataclass

import openai_manager
from openai_manager.utils import timeit

from cache import cache_in_file, cache_two_string_results
from chunking import Chunk


@dataclass
class Idea:
  alternative_titles: List[str]
  title: str
  body: str
  origin_chunks: List[Chunk]


@cache_in_file
def chunks_to_propositions(
    chunks: List[Chunk]) -> List[Tuple[Chunk, List[str]]]:
  prompt = "Suggest a non-trivial proposition (a sentence no " \
  "longer than 5-10 words) which could be extracted from the following " \
  "piece of text. The proposition shouldn't necessarily be the sole " \
  "theme of the text. The proposition should be specific rather " \
  "than general or abstract. The proposition should describe a single " \
  "concept or compare it to another concept.\n\n" \
  "Text:\"{}\"\n\nProposition:"

  # Rate limiting within a batch is not yet correctly implemented
  # in openai_manager library, see
  # https://github.com/MrZilinXiao/openai-manager/blob/fdd7121a1/openai_manager/producer_consumer.py#L106,
  # So we don't take advantage of parallel async requests yet
  openai_responses = []
  for chunk in chunks:
    messages = [
      {
        "role": "user",
        "content": prompt.format(chunk.chunk)
      },
    ]
    openai_responses += openai_manager.ChatCompletion.create(
      model="gpt-4",
      messages=messages,
      # 1.0 (default) gives too repetitive responses
      temperature=1.3,
      n=15)

  chunks_and_propositions = []
  for chunk, response in zip(chunks, openai_responses):
    try:
      propositions = [c["message"]["content"] for c in response["choices"]]
      # set() for deduplication of propositions after cleaning
      propositions = sorted(list(set(map(clean_proposition, propositions))))
      chunks_and_propositions.append((chunk, propositions))
    except Exception as e:
      print(
        f'Unexpected response {response} to proposition-eliciting prompt on "{chunk}"'
      )
      print(e)
  return chunks_and_propositions

def clean_proposition(p: str) -> str:
  p = p.strip()
  if p.startswith("Proposition: "):
    p = p[len("Proposition: "):]
  p = p.strip('"')
  if p.endswith('.'):
    p = p[:-1]
  # Replace all double quotes with single quotes within the proposition to make the future
  # life easier when propositions are wrapped in double-quotes in various prompts.
  p = p.replace('"', "'")
  return p

@cache_in_file
@timeit
def propositions_to_ideas(
    chunks_and_propositions: List[Tuple[Chunk, List[str]]]) -> List[List[Idea]]:
  ideas_per_chunk = []
  for chunk, propositions in chunks_and_propositions:
    clusters = cluster_propositions(propositions)
    ideas = []
    for cluster in clusters:
      idea = Idea(alternative_titles=cluster,
                  title=cluster[0],
                  body="",
                  origin_chunks=[chunk])
      ideas.append(idea)
    ideas_per_chunk.append(ideas)
  return ideas_per_chunk

def cluster_propositions(propositions: List[str]) -> List[List[str]]:
  """
  Cluster propositions in the groups that are re-phrasings/reformulations of the same
  idea, rather than express diffirent ideas (even if related), as per
  assess_same_idea() function.

  assess_same_idea() is *not* guaranteed to produce perfectly cliqued clusters of props
  where for each pair within the cluster the result of assess_same_idea() is "Yes" and
  for any pair with one prop from the cluster and one prop that is not in the cluster
  the result is "No".

  "Properly" the algorithm should call assess_same_idea() for all props pairwise and
  then run an approximate clique finding algorithm on the resulting graph (with half-edges,
  because assess_same_idea() could also return "Maybe"), but this may be too expensive
  because each assess_same_idea() call uses GPT-4.

  Instead, cluster_propositions() currently implements a simple greedy algorithm,
  finding connected subgraphs rather than cliques, disregarding potential noisiness of
  assess_same_idea().

  TODO a relatively easy to implement idea to improve this algorithm is to try to split
       the resulting clusters in two using embedding similarity of the propositions in
       the cluster. Or just use nltk.metrics.distance.
  """
  # A list to store clusters. Each cluster is a list of propositions.
  clusters = []

  for prop in propositions:
    found_cluster = False
    maybe_cluster_index = None

    for i, cluster in enumerate(clusters):
      # Check the assessment result with one prop from the current cluster
      is_same_idea = assess_same_idea(prop, cluster[0])

      if is_same_idea == "Yes":
        cluster.append(prop)
        # Sort the props in the cluster by length, so that the shortest prop
        # from the cluster is used in future comparisons. This might be better
        # because shorter props are more contrast-y than longer ones.
        cluster.sort(key=len)
        found_cluster = True
        break
      elif is_same_idea == "Maybe" and maybe_cluster_index is None:
        # Currently, just consider the first "Maybe" cluster encountered
        # and ignore the rest.
        # TODO more advanced handling of "Maybe" is possible: call
        # assess_same_idea() for the prop and other elements of the cluster,
        # to distinguish "Maybe + Yes" and "Maybe + No" clusters.
        maybe_cluster_index = i

    if not found_cluster:
      if maybe_cluster_index is not None:
        # If there was a "Maybe" result and no "Yes" result,
        # add the prop to the (first) cluster that returned "Maybe"
        clusters[maybe_cluster_index].append(prop)
      else:
        # If there was neither a "Yes" nor a "Maybe" result, create a new cluster
        clusters.append([prop])

    # Re-sort clusters in reverse size order to reduce the number of assess_same_idea()
    # calls for subsequent props (bigger clusters will tend to grow bigger still)
    clusters.sort(key=len, reverse=True)
      
  return clusters


@cache_two_string_results
@timeit
def assess_same_idea(s1: str, s2: str) -> str:
  # TODO: Increase the repertoire of few-shot examples and maybe even use
  #       ensemble assessment, i.e., still query with temperature 0, but
  #       on two-three different sets of few-shot examples to check that
  #       the answer of the LLM stays the same regardless of these examples.
  #       This also requires clearing our mind about what do we mean by
  #       "sameness or distinctness of ideas", semantically and syntactically,
  #       and cover the categories of "sameness/distinctness" that are not yet
  #       covered in the examples below.
  # TODO: Haven't checked if this prompt ever returns "Maybe/unclear". Also,
  #       the example given for "Maybe/unclear" might not be even a good example,
  #       maybe it should rather be "No".
  prompt = '''Do these two sentences express essentially the same idea? Answer either "Yes", "Maybe/unclear", or "No".

S1: "Evergreen notes should be atomic"
S2: "Evergreen notes should be densely linked"
A: No

S1: "Negatively valenced states lead to reduced reliance on prior expectations."
S2: "Valence influences action selection through confidence in internal models."
A: Maybe/unclear

S1: "Lithium insertion into graphite releases heat."
S2: "Lithium intercalation into graphite is exothermic"
A: Yes

S1: "{}"
S2: "{}"
A:'''.format(s1, s2)
  r = openai_manager.ChatCompletion.create(
    # GPT-4 is needed here, GPT-3.5 often returns a wrong answer
    model="gpt-4",
    messages=[
      {
        "role": "user",
        "content": prompt
      },
    ],
    temperature=0)
  answer = r[0]["choices"][0]["message"]["content"]
  # Drop "/unclear" part from "Maybe/unclear" answer
  return answer.split("/", 1)[0]
