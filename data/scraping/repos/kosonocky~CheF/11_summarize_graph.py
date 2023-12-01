import time
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
    
    user_prompt_complete = user_prompt.replace("__INSERT_DESCRIPTORS_HERE__", descriptors)
    
    # ensure less than 4k tokens. Most are ~1k
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
    # print(f"INFO: Using {(gpt_model:='gpt-3.5-turbo')} for model")
    print(f"INFO: Using {(gpt_model:='gpt-4')} for model")
    print(f"INFO: Using {(n_cpus:=8)} CPUs")

    api_key = ""
    gpt_system_prompt = "You are a PhD pharmaceutical chemist"
    gpt_user_prompt = f"""
        Pretend you are a pharmaceutical chemist. I will provide you with several terms, and your job is to summarize the terms into appropriate categories. Be succinct, focusing on the broadest categories while still being representative. Don't show your work.

        Example terms:
        Antiviral
        HCV
        Kinase
        Cancer
        Polymerase
        Protease


        Example summarization:
        Antiviral & Cancer


        Terms:

        __INSERT_DESCRIPTORS_HERE__

        Summarization:
        """ 
    
    # save prompt
    with open(f"graph_summarization_prompt.txt", "w") as f:
        f.write(gpt_system_prompt + "\n\n" + gpt_user_prompt)
    
    df = pd.read_csv("gephi_graphs/modularity_classes.csv")
    
    summ_1 = []
    for i in range(19):
        print(i)
        labels = df.loc[df["Modularity Class"] == i]["Label"].values
        # convert labels to string separated by newlines
        labels = "\n".join(labels)
        summ = label_summarization_wrapper(labels, api_key, gpt_model, gpt_system_prompt, gpt_user_prompt, gpt_temperature)
        summ_1.append(summ)
    
    

    # NOTE unused in final analysis. Did not work as well as expected.
    gpt_user_prompt_2 = f"""{gpt_user_prompt.replace("__INSERT_DESCRIPTORS_HERE__", "")}
    __INSERT_DESCRIPTORS_HERE__

    Summarize these terms further, but ensure categories are still highly specific.

    Summarization:
    """

    summ_2 = []
    for i in range(19):
        print(i)
        summ = label_summarization_wrapper(summ_1[i], api_key, gpt_model, gpt_system_prompt, gpt_user_prompt_2, gpt_temperature)
        summ_2.append(summ)
    

    output_df = pd.DataFrame({"summ_1": summ_1, "summ_2": summ_2})
    output_df.to_csv("gephi_graphs/gpt_summarized_labels.csv", index=False)
    

if __name__ == "__main__":
    main()