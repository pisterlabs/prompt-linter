import pandas as pd
import openai
import subprocess
import json
    
def generate_resources(df):
    # gnerating the new prompt with resources
    print("Generating the new prompt with resources...")
    autocompletion_with_resources = []
    for index, row in df.iterrows():
        autocompletion = row["response_txt"].strip()
        link = row["link"].strip()
        new_autocompletion =  f"{autocompletion}\\nAyrıca bu konuyla ilgili daha fazla bilgi için şu videomuza da göz atabilirsiniz:\\n{link}"
        new_autocompletion = new_autocompletion.strip()
        autocompletion_with_resources.append(new_autocompletion)
    
    for index in range(len(df)):
        # df.loc[index, ["sub_prompt"]] = df.loc[index, ["sub_prompt"]] + "\n\n###\n\n"
        df.loc[index, ["sub_prompt"]] = df.loc[index, ["sub_prompt"]]
    
    df["autocompletion_with_resources"] = autocompletion_with_resources
    # print(df.columns)

def preprocess_dataset(input_dataset_name, final_dataset_name):
    # reading the dataset
    print("Starting to reading the dataset...")
    df = pd.read_excel(f"{input_dataset_name}")
    
    # processing the dataset
    generate_resources(df)

    # prepared_data = df.loc[:,['sub_prompt','autocompletion_with_resources']]
    df = df.rename(columns = {'sub_prompt':'prompt', 'autocompletion_with_resources':'completion'})
    df.to_excel(f'../datasets/final.xlsx', index=False)

    df.drop(['age', 'gender', "activity", "prompt", "response_txt", "finish_reason", "link"], axis = 1, inplace = True) 
    prepared_data = df.copy()
    prepared_data = prepared_data.rename(columns = {'sub_prompt':'prompt', 'autocompletion_with_resources':'completion'})
    
    # prepared_data.rename(columns={'sub_prompt':'prompt', 'autocompletion_with_resources':'completion'}, inplace=True)
    prepared_data.to_excel(f'{final_dataset_name}', index=False)
    
    print(prepared_data)
    
    final_dataset_name_for_json = final_dataset_name[:].replace(".csv", ".json")
    json_str = prepared_data.to_json(orient='records')
    json_object = json.loads(json_str)
    # print(json_object)
    
    with open(f'{final_dataset_name_for_json}', 'w') as json_file:
        json.dump(json_object, json_file)
        
    # with open(f'{final_dataset_name_for_json.replace("json", "jsonl")}', 'w') as _jsonl_file:
    #     for index, row in prepared_data.iterrows():
    #         to_write = "{" + f"{row['prompt']}, {row['completion']}" + "}\n"
    #         _jsonl_file.write(to_write)
    
    print(f"Saved to {final_dataset_name}")

def main(input_dataset_name, final_dataset_name):
    ## preprocess_dataset
    preprocess_dataset(input_dataset_name=input_dataset_name, final_dataset_name=final_dataset_name)

    
if __name__ == "__main__":
    input_dataset_name = "../datasets/generated_dataset.xlsx" 
    final_dataset_name = "../datasets/generated_dataset_preprocessed.xlsx"
    
    main(final_dataset_name=final_dataset_name, input_dataset_name=input_dataset_name)