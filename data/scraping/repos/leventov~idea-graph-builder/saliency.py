from typing import List
import re

import openai_manager
from openai_manager.utils import timeit

from cache import cache_in_file
from ideas import Idea, clean_proposition

@cache_in_file
def only_salient_ideas(ideas_per_chunk: List[List[Idea]]) -> List[Idea]:
  all_ideas = []
  for ideas in ideas_per_chunk:
    largest_cluster_prop = max(ideas, key=lambda i: len(i.alternative_titles)).title
    props_by_saliency = order_propositions_by_saliency([i.title for i in ideas])
    most_salient_props = props_by_saliency[:5]
    if largest_cluster_prop not in most_salient_props:
      most_salient_props = most_salient_props[:4] + [largest_cluster_prop]
    prop_to_idea = {i.title: i for i in ideas}
    for prop in most_salient_props:
      all_ideas.append(prop_to_idea[prop])
  return all_ideas

@timeit
def order_propositions_by_saliency(propositions: List[str]) -> List[str]:
  question = "Order the following propositions in the order from more original, " \
    "salient, and specific to more generic, common-sensical, abstract, banal, and cluttered." \
    "Shortly explain the position of the sentence in parens after it:\n"
  for idx, prop in enumerate(propositions, 1):
    question += f"{idx}. \"{prop}.\"\n"

  response = openai_manager.ChatCompletion.create(
    model="gpt-4",
    messages=[
      {"role": "user", "content": question},
    ],
    temperature=0)

  response_content = response[0]['choices'][0]['message']['content']

  # Finding the first occurrence of "1." to start extracting the sentences
  start_index = response_content.find("1.")

  # Extracting and cleansing the ordered sentences
  ordered_propositions = []
  for line in response_content[start_index:].split('\n'):
    if line.strip() == '':
      continue
    # Remove the parenthetical explanation, LLM will definitely not nest any
    # parens within this explanation, parens are not in LLM's style of writing.
    if line.rfind('(') > 0:
      line = line[:line.rfind('(')]
    else:
      # Sometimes the LLM provides explanation after a hypthen instead of placing
      # it in parens.
      line = line[:line.rfind('. -')]

    no_number_line = re.sub(r'^\d+\.', '', line)
    # Break if the list has ended: the line which doesn't start with number reached.
    if no_number_line == line:
      break
    else:
      line = no_number_line
    ordered_propositions.append(clean_proposition(line))
  # TODO the LLM may slightly rephrase the propositions when sorting them, e.g.,
  #      if the proposition containes an unusual word or phrase spelling, such as
  #      "fixed-size" rather than "fixed-sized", and the LLM cannot help but to
  #      spell it in the more common way in the ordered list of propositions. To
  #      circumvent this problem, we should compute embeddings for the original
  #      propositions and the sentences from the ordered list and do pair the closest.
  raise "fixme"
  return ordered_propositions