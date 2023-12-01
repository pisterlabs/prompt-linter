import cohere
co = cohere.Client('')

# example prompt
prompt="Where do the tallest penguins live?"
response = co.chat(
    message = prompt,
    connectors = [{"id": "web-search"}], 
    prompt_truncation="AUTO"
)

docs_used = [citation['document_ids'] for citation in response.citations]
docs_used = [item for sublist in docs_used for item in sublist]
matched_urls = [doc['url'] for doc in response.documents if doc['id'] in docs_used]
matched_titles = [doc['title'] for doc in response.documents if doc['id'] in docs_used]
print(matched_urls)
print(matched_titles)
