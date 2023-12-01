import openai
import os
import json

OPENAI_API_KEY = "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"
SMART_LLM_MODEL = "gpt-4"
openai.api_key = OPENAI_API_KEY
cwd = os.getcwd()
OUTPUT_FILE = cwd + "/graph.json"

def articulate_similarity(list_of_highlight: list):
    '''
    ariculate the similarity among the list of highlight
    '''

    temperature = 1.0

    output_format = 'a sentence less than 40 chracters'
    none_handle = "I couldn't find any similarity"
    temperature = 0.4

    prompt = f"find the similar concept among these list of text per {list_of_highlight}. For the output, please follow this format: {output_format}. If you couldn't find any similarity, please output {none_handle}"


    result = openai.ChatCompletion.create(
        model=SMART_LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=200,
        top_p=1.0,
        frequency_penalty=0.8,
        presence_penalty=0.0
        )

    '''
    #To-Do: handle the none
    if "None" in result.choices[0].message["content"]:
        print(f"No {connection_type} found in {content}")
        return []
    '''

    #print(result.choices[0].message["content"])
    #example output: '["Quick Passages", "20VC", "Alice Zong"]'

    #To-Do: output handler

    result = result.choices[0].message["content"]

    print(result)

    #save to file
    with open(OUTPUT_FILE,'r') as file:
        graph = json.load(file)

    graph["edges"]["description"] = result

    # Save the updated Database 1 to a json file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(graph, f)

    return result

def main():
    with open(OUTPUT_FILE,'r') as file:
         graph = json.load(file)

    for edge in graph["edges"]:
        descripion =  articulate_similarity(edge)
        graph["edges"][edge]["description"] = descripion

    with open(OUTPUT_FILE, "w") as f:
        json.dump(graph, f)

    return


#For debug only
if __name__ == '__main__':
    main()

