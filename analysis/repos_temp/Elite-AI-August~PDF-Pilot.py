"""Use the following pieces of context to answer the users question.
Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
----------------
{summaries}"""f"""### Question: 
    {query}
    ### Answer: 
    {result['answer']}
    ### Sources: 
    {result['sources']}
    ### All relevant sources:
    {' '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))}
    """