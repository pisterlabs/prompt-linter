import json
import openai
import promptStorage as prompts
import embeddingSimilarity
import time

def main():
    pass

def searching_stage(similar_queries_list):
    print("Starting search stage...")
    similar_content_rows = []
    legal_text_list = []
    legal_text_tokens_list = []

    print("  - Searching relevant sections for lawful template")
    begin = time.time()
    lawful = search_similar_content_sections(similar_queries_list[0], matches=40)
    legal_text, legal_text_tokens_l = accumulate_legal_text_from_sections(lawful, used_model="gpt-3.5-turbo-16k")
    legal_text_lawful, citation_list = embeddingSimilarity.format_sql_rows(legal_text)
    end = time.time()
    print("    * Total time for vector similarity: {}".format(round(end-begin, 2)))

    '''
    print("  - Searching relevant sections for unlawful template")
    begin = time.time()
    unlawful = search_similar_content_sections(similar_queries_list[4], matches=40)
    legal_text, legal_text_tokens_u = accumulate_legal_text_from_sections(unlawful, used_model="gpt-3.5-turbo-16k")
    legal_text_unlawful = embeddingSimilarity.format_sql_rows(legal_text)
    end = time.time()
    print("    * Total time for vector similarity: {}".format(round(end-begin, 2)))
    '''
    legal_text_tokens_list = [legal_text_tokens_l, legal_text_tokens_l, legal_text_tokens_l, legal_text_tokens_l, legal_text_tokens_l]
    similar_content_rows = [lawful, lawful, lawful, None, None]
    legal_text_list = [legal_text_lawful,legal_text_lawful,legal_text_lawful,None, None]

    return similar_content_rows, legal_text_list, legal_text_tokens_list, citation_list

def search_similar_content_sections(modified_user_query, matches=20):
    
    # Get cosine similarity score of related queries to all content embeddings
    return embeddingSimilarity.compare_content_embeddings(modified_user_query, match_count=matches)

def accumulate_legal_text_from_sections(sections, used_model):
    current_tokens = 0
    row = 0
    legal_text = []
    used_model = "gpt-3.5-turbo-16k"
    if used_model == "gpt-4-32k":
        max_tokens = 24000
    elif used_model == "gpt-4":
        max_tokens = 5000
    elif used_model == "gpt-3.5-turbo-16k":
        max_tokens = 12000
    elif used_model == "gpt-3.5-turbo":
        max_tokens = 2000
    max_tokens = 24000
    while current_tokens < max_tokens and row < len(sections):
        #print(sections[row])
        current_tokens += sections[row][12]
        legal_text.append(sections[row])
        row += 1
    return legal_text, current_tokens



if __name__ == "__main__":
    main()