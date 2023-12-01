# text -> extract subject -> search wikipedia -> semantic search text -> identify citations

import toml
import cohere
import numpy as np

from subject import SubjectExtractor
import wiki


config = toml.load("config.toml")

co = cohere.Client(config['cohere']['key'])

subject_extractor = SubjectExtractor(co)


claim = """
The sigmoid function can lead to the vanishing gradient problem.
""".strip()

claim_embed = np.array(co.embed([claim]).embeddings)

subject = subject_extractor.query(claim)
if ";" in subject:
    subject = subject.split(";")[0]

pages = wiki.wikipedia.search(subject)


merged_texts = []
merged_citations = []
merge_citation_map = {}

for page in pages[:3]:
    soup = wiki.fetch_page(page)
    pairs, citations = wiki.extract_text_with_citations(soup)

    merged_texts += [pair[0] for pair in pairs]
    merged_citations += [[page+x for x in pair[1]] for pair in pairs]
    merge_citation_map.update({page+x: citations[x] for x in citations})

embeddings = np.array(co.embed(merged_texts).embeddings)

scores = np.dot(embeddings, claim_embed.T).flatten()
scores = scores / np.linalg.norm(embeddings, axis=1)
scores = scores / np.linalg.norm(claim_embed)

for i in np.argsort(scores)[::-1][:3]:
    print("="*40)
    print(scores[i])

    print(merged_texts[i])
    print("-"*40)

    for citation in merged_citations[i]:
        print(merge_citation_map[citation])
    
    
