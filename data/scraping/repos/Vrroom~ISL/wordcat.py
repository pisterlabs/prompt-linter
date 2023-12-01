import openai
import pdb
import json
import argparse
import isl_utils as islutils
import pandas as pd
import os

GPT_MODEL = "gpt-4"
ARG_HELP = "[All|Words|Categorise|Map] All (default) - run all steps; Categorise - categorises the words using chatgpt and saves in wordtext.json file; Words - generate video_title.txt file by analysing the metadata files and retrieving the video titles; Map - puts the word and their categories back into the meta file"

def initialise_tables(video_json_dir) :
    hash_metadata_frame = pd.read_csv(islutils.VIDEO_HASH_METADATA)
    video_metadata_frame = pd.read_csv(islutils.VIDEO_METADATA)
    metadata_table = pd.merge(hash_metadata_frame, video_metadata_frame, on="hash", how="inner")
    
    for index, row in metadata_table.iterrows() :
        video_path = row['path']
        filename, _ = os.path.splitext(os.path.basename(video_path))
        jsonfile = video_json_dir +  "/" + filename + ".json"
        video_title = ""
        processed = False
        if (os.path.exists(jsonfile)) :
            with open(jsonfile, 'r') as f:
                data = json.load(f)
                video_title = data['title']
                processed = True
        metadata_table.at[index, "Processed"] = processed
        metadata_table.at[index, "Title"] = video_title

    return metadata_table

def gen_word_categories(words_to_categorize, batch, start_index) :
    categorized_words = {}
    i = start_index    
    arr_len = len(words_to_categorize)

    while i < arr_len :
        # Create a prompt for GPT-3 to assist with categorization
        print ("Starting " + str(i) + " to " + str(i+batch))
        small_list = words_to_categorize[i:i+batch]
        prompt = f"Fix typos in the following words, remove unnecessary suffixes like -1 or -2 and then categorise them based on what you think makes best sense. Please provide more than one category if the word can belong to more than one category since no context is available. Put the words that need more context under miscellaneous category. And do not provide any explanations or any notes. Nor do put a running count when outputing the categories, nor prefix the category with Category. And finally result a python dictionary where the key is the word and value is the list of categories that apply to that word.:  \nWords: {', '.join(small_list)}."
#        prompt = f"Categorize the following words into the appropriate categories. it is fine to have a word in multiple categories. In case the passed list of categories is insufficient to properly categorise the word, please categorise as you see fit. Ignore the words that you cannot categorise well: \nWords: {', '.join(small_list)}.\nCategories: {', '.join(categories)}."
        try:
            response = openai.ChatCompletion.create( 
                model=GPT_MODEL,  # You can adjust the engine based on your requirements
                messages=[
                    {'role': 'system', 'content': 'You classify words into categories'},
                    {'role': 'user', 'content': prompt},
                ],
            )
            resp = response['choices'][0]['message']['content']
            categorized_words.update(json.loads(resp))
            
            # if not resp.startswith('Note:') :
            #     categorized_words = categorized_words + "\n" + resp
            # Extract the categorized words from the response
            # categorized_words = response['choices'][0]['message']['content']
            # final_json = final_json.update(json.loads(categorized_words))

            print ("Done " + str(i) + " to " + str(i+batch))
            i = i + batch
            if arr_len - i < batch :
                batch = arr_len - i            
        except Exception as e:
            print ("exception occured. Retrying ")
            print (e)
            continue
    
    return categorized_words

def extract_titles(metadata_table) :
    # save titles in a file
    print("doing words!")
    titles = []
    processed_rows = metadata_table[metadata_table['Processed'] == True]
    print(len(processed_rows))
    for index, row in processed_rows.iterrows() :
        titles.append(row['Title'])

    with open ("./video_title.txt", 'w') as video_title_file :
        for item in titles :
            video_title_file.write(item + "\n")

def categorise_titles() :
    # List of English words you want to categorize
    with open("./video_title.txt", 'r') as wordfile :
        words_to_categorize = [line.strip() for line in wordfile.readlines() if line.strip()]
    categorized_words = gen_word_categories(words_to_categorize, 100, 0)
    
    with open('wordcat.json', 'w') as f:
        json.dump(categorized_words, f, ensure_ascii=False)

def update_video_metadata(metadata_table) :
    with open('./wordcat.json', 'r') as file:
        json_data = file.read()

    json_data = json_data.replace("'", "\"")
    categorized_words = json.loads(json_data)

    for title, cat in categorized_words.items() :
        rows = metadata_table[metadata_table['Title'] == title]
        catStr = ', '.join(cat)
        for i in range(len(rows)) :
            hashvalue = rows.iloc[i]['hash']
            cond = metadata_table['hash'] == hashvalue
            row_index = metadata_table.index[cond].tolist()[0]
            metadata_table.at[row_index, "Categories"] = catStr

    metadata_table.to_csv('./video_metadata_categorised.csv', index = False)


def main (video_json, step) :
    step = step.lower()
    doCategorise = step == 'categorise'
    doWord = step == 'words'
    doMap = step == 'map'
    if step == 'all' :
        doCategorise = doWord = doMap = True
        return

    import pdb
    metadata_table = initialise_tables(video_json)

    if doWord :
        print("Extracting titles")
        extract_titles(metadata_table)
        print("done")

    if doCategorise :
        print("Doing Categorisation")
        categorise_titles()
        print("done")


    if doMap:
        print("Doing Mapping")
        update_video_metadata(metadata_table)
        print("done. Generated video_metadata_categorised.csv in the current directory")

if __name__ == "__main__" : 

    parser = argparse.ArgumentParser(description="Update video metadata file.")
    parser.add_argument('--key', required=True, type=str, help="Open API Key")
    parser.add_argument('--video_json_dir', required=True, type=str, help="Path to the video json directory.")
    parser.add_argument('--step', required=False, default="all", type=str, help=ARG_HELP)
    args = parser.parse_args()

    openai.api_key = args.key
    main(args.video_json_dir, args.step)

