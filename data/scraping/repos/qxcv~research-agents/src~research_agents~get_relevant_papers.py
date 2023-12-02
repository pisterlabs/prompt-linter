import anthropic
from research_agents.prophy_scrape import download_prophy

def get_relevant_papers(client, title_original, abstract_original, title_refs, abstract_refs, stop_len=1):
    """
    Given a reference title and abstract, and two lists of paired titles and abstracts, find the first "stop_len" relevant papers.
    """

    # Loop through the comparisons
    good_idxs = []
    for i, (title_ref, abstract_ref) in enumerate(zip(title_refs, abstract_refs)):
        resp = client.completion(
                prompt=f"{anthropic.HUMAN_PROMPT} You have two papers. The first paper title is: {title_original}, the first paper abstract is: {abstract_original}. The second paper title is: {title_ref}, the second paper abstract is: {abstract_ref}. Are these two papers related? Just answer yes or no. {anthropic.AI_PROMPT}",
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model="claude-1",
                max_tokens_to_sample=100,
                temperature=0
            )
        if resp["completion"].lower().strip().startswith("yes"):
            good_idxs.append(i)

        if len(good_idxs) == stop_len:
            break

    return good_idxs

if __name__ == "__main__":

    # Initialize the client
    client = anthropic.Client(api_key="sk-ant-api03-FILL-IN-THIS-KEY")

    url = 'https://arxiv.org/abs/2004.04136'
    
    title_original, abstract_original, citing_data = download_prophy(url)
    print(f'Original title: {title_original}')
    print('='*80)
    print(f'Original abstract: {abstract_original}')
    print('='*80)
    title_refs = [d['title'] for d in citing_data]
    abstract_refs = [d['abstract'] for d in citing_data]
    print(f'Num citations: {len(title_refs)}')
    print(f'Title references: {title_refs}')

    # Declare max number of papers to return
    stop_len = 3

    # Get the good indices
    good_idxs = get_relevant_papers(client, title_original, abstract_original, title_refs, abstract_refs, stop_len)

    print("Good indices: ", good_idxs)


