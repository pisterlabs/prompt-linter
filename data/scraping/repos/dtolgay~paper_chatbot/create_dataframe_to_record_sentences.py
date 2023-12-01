def create_dataframe_to_record_sentences_and_embeddings(
                                                        pages,
                                                        EMBEDDING_MODEL="text-embedding-ada-002",
                                                        do_you_want_to_save_to_csv_file=0
                                                        ):

    import openai
    import pandas as pd
    import nltk 

    df = pd.DataFrame(columns=["Page", "Sentence"])

    sentences_list = []
    embeddings_list = []

    # for current_page_number in range (0, len(pages)):
    for current_page_number in range (1, 2):
        page_content = pages[current_page_number].page_content
        sentence_content = nltk.sent_tokenize(page_content)
        for current_sentence in range (0, len(sentence_content)):
            # Seperating into different sentences
            sentences_list.append(sentence_content[current_sentence])
            
            # Finding the embeddings for the sentences sepereted above
            response = openai.Embedding.create(model=EMBEDDING_MODEL, input=sentence_content[current_sentence])
            embeddings = response['data'][0]['embedding']
            embeddings_list.append(embeddings)        
    

    # Assuming sentences_list and embeddings_list have the same length
    data = {'sentences': sentences_list, 'embeddings': embeddings_list}
    df = pd.DataFrame(data)
    df["sentences"] = df["sentences"].str.replace("\n", " ")

    if (do_you_want_to_save_to_csv_file==1):
        df.to_csv('sentences_and_embeddings.csv', index=False)
        print("csv file is saved!")

    return df