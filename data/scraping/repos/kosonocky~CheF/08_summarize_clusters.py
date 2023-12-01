# NOTE The largest cluster is (likely always) a structural term cluster. This one results in API REQUEST ERROR, and will be manually fixed to 'structural' in the final results, which will be removed from the final results.

import time
import multiprocessing as mp
from itertools import repeat

import pandas as pd

import openai
import backoff 


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def chatgpt(api_key, model = "gpt-3.5-turbo", system_prompt = "", user_prompt = "", gpt_temperature=0):
    """
    Uses OpenAI GPT Model to generate text.
    
    Parameters:
    -----------
    api_key : str
        OpenAI API key.
    
    model : str
        OpenAI model to use.
    
    system_prompt : str
        System prompt for OpenAI model.
    
    user_prompt : str
        User prompt for OpenAI model.

    gpt_temperature : float
        Temperature for OpenAI model.
    
    Returns:
    --------
    return : str
        Generated text.
    """
    if (user_prompt == "") or (system_prompt == "") or (api_key == "") or (model == ""):
        print("MISSING PROMPT OR API KEY OR MODEL")
        return ""
    try:
        openai.api_key = api_key
        response = completions_with_backoff(
            model=model,
            temperature = gpt_temperature,
            messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(e)
        return "API REQUEST ERROR"

def label_summarization_wrapper(descriptors, api_key, model = "gpt-3.5-turbo", system_prompt = "", user_prompt = "", gpt_temperature=0):       
    """
    Wrapper function for chatgpt label set summarization.

    Summarizes set of labels associated with a given molecule into a single label. Uses OpenAI GPT Model.

    Parameters:
    -----------
    descriptors : str
        Comma-separated descriptors. If only one descriptor, only contains that string.

    api_key : str   
        OpenAI API key.

    model : str
        OpenAI model name.

    system_prompt : str
        System prompt for OpenAI model.

    user_prompt : str
        User prompt for OpenAI model.

    gpt_temperature : float 
        Temperature for OpenAI model.

    Returns:
    --------
    summ_lable : str
        Label summarization
    """
    
    if len(descriptors) == 0: # abort if nothing to summarize
        return descriptors
    
    if "," not in descriptors: # abort if no commas found in descriptor
        return descriptors
    
    user_prompt_complete = user_prompt.replace("__INSERT_DESCRIPTORS_HERE__", descriptors)
    

    # Unused code to ensure less than 4k tokens. Most are ~1k
    # enc = tiktoken.get_encoding("cl100k_base")
    # tok = enc.encode(user_prompt_complete)
    # user_prompt_complete = enc.decode(tok[:4000])
    
    summ_label = chatgpt(api_key = api_key,
                    model = model,
                    system_prompt = system_prompt,
                    user_prompt = user_prompt_complete,
                    gpt_temperature = gpt_temperature,)
    if summ_label == "API REQUEST ERROR":
        print(f"API REQUEST ERROR for {descriptors}")
    return summ_label

def main():
    t_curr = time.time()
    print(f"INFO: Using {(gpt_temperature:=0)} temperature for GPT")
    print(f"INFO: Using {(gpt_model:='gpt-3.5-turbo')} for model")
    print(f"INFO: Using {(n_cpus:=8)} CPUs")

    fname = "eps_0.340_diff_20030_clusters"
    save_path_template = f"../results/schembl_summs_v4_vocab_gpt_cleaned_{fname}"

    api_key = ""
    gpt_system_prompt = "You are a PhD pharmaceutical chemist"
    gpt_user_prompt = f"""
        Given a set of molecular descriptors, return a single descriptor representing the centroid of the terms. Do not speculate. Only use the information provided. Be concise, not explaining answers.

        Example 1 Set of Descriptors:
        11(beta)-hsd1, 11-hsd-2, 17β-hsd3

        Example 1 Average Descriptor:
        hsd

        Example 2 Set of Descriptors:
        anti-retroviral, anti-retrovirus, anti-viral, anti-virus, antiretroviral, antiretrovirus, antiviral, antivirus

        Example 2 Average Descriptor:
        antiviral

        Set of Descriptors:
        __INSERT_DESCRIPTORS_HERE__

        Average Descriptor:
        """ 
    
    # save prompt
    with open(f"{save_path_template}_prompt.txt", "w") as f:
        f.write(gpt_system_prompt + "\n\n" + gpt_user_prompt)
    
    # read in data
    with open(f"../results/eps_diffs/{fname}.txt", "r") as f:
        clusters = f.read().splitlines()

    print(clusters)
    print(f"Time to read in data: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds.")

    with mp.Pool(n_cpus) as p:
        new_labels = p.starmap(label_summarization_wrapper, zip(clusters, repeat(api_key), repeat(gpt_model), repeat(gpt_system_prompt), repeat(gpt_user_prompt), repeat(gpt_temperature)))
        print(f"Time to get patent info: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds.")
        
    df = pd.DataFrame({"original_clustered_labels": clusters})
    df["gpt_cleaned_labels"] = new_labels
    df[["original_clustered_labels", "gpt_cleaned_labels"]].to_csv(f"{save_path_template}.csv", index=False)

if __name__ == "__main__":
    main()