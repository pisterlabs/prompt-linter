import os
import argparse
from transformers import AutoTokenizer
import numpy as np
import openai

openai.organization = None
openai.api_key = None

def parse_args():
    parser = argparse.ArgumentParser(description='Calculates surprisal and other \
                                    metrics (in development) of transformers language models')

    parser.add_argument('--stimuli', '-i', type=str,
                        help='Stimuli to test.')
    parser.add_argument('--stimuli_list', '-ii', type=str,
                        help='Path to file containing list of stimulus files to test.')
    parser.add_argument('--output_directory','-o', type=str, required = True,
                        help='Output directory.')
    parser.add_argument('--model','-m', type=str,
                        help='The name of the GPT-3 model to run.')
    parser.add_argument('--model_list','-mm', type=str,
                        help='Path to file with a list of GPT-3 models to run.')
    parser.add_argument('--key','-k', type=str,
                        help='Your OpenAI API key.')

    args = parser.parse_args()
    return args

def process_args(args):
    
    try:
        output_directory = args.output_directory
    except:
        print("Error: Please specify a valid output directory.")

    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
        except:
            print("Error: Cannot create output directory (Note: output directory does not already exist).")
    
    if args.model_list:
        try:
            assert os.path.exists(args.model_list)
            with open(args.model_list, "r") as f:
                model_list = f.read().splitlines()
        except:
            print("Error: 'model_list' argument does not have a valid path. Trying to use individual specified model.")
            try:
                assert args.model
                model_list = [args.model]
            except:
                print("Error: No model specified")
    else:
        try:
            assert args.model
            model_list = [args.model]
        except:
            print("Error: No model specified") 
            
            
    if args.stimuli_list:
        try:
            assert os.path.exists(args.stimuli_list)
            with open(args.stimuli_list, "r") as f:
                stimulus_file_list = f.read().splitlines()
        except:
            print("Error: 'stimuli_list' argument does not have a valid path. Trying to use individual stimulus set.")
            try:
                assert args.stimuli
                stimulus_file_list = [args.stimuli]
            except:
                print("Error: No stimuli specified")
    else:
        try:
            assert args.stimuli
            stimulus_file_list = [args.stimuli]
        except:
            print("Error: No stimuli specified")  

    try:
        if openai.api_key==None:
            assert args.key
            openai.api_key = args.key

    except:
        print("No API Key. Unable to run GPT-3.")   

                
    return(output_directory,model_list,stimulus_file_list)  

def run_models(output_directory,model_list,stimulus_file_list):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    for j in range(len(model_list)):
        model_name = model_list[j]
        model_name_cleaned = model_name.replace("-","_")
        for i in range(len(stimulus_file_list)):
            stimuli_name = stimulus_file_list[i].split('/')[-1].split('.')[0] 
            filename = output_directory + "/" + stimuli_name + "." + "surprisal" + "." + model_name_cleaned + ".causal.output"
            with open(filename,"w") as f:
                f.write("FullSentence\tSentence\tTargetWords\tSurprisal\tNumTokens\n")
            
            with open(stimulus_file_list[i],'r') as f:
                stimulus_list = f.read().splitlines() 
            for j in range(len(stimulus_list)):
                try:
                    stimulus = stimulus_list[j]
                    stimulus_spaces = stimulus.replace("*", "α")
                    stimulus_spaces = stimulus_spaces.replace(" α", "α ")
                    encoded_stimulus = tokenizer.encode(stimulus_spaces)

                    if (len(tokenizer.tokenize("aα"))==2): 
                        dummy_var_idxs = np.where((np.array(encoded_stimulus)==tokenizer.encode("α")[-1]) | (np.array(encoded_stimulus)==tokenizer.encode("aα")[-1]))[0]
                        preceding_context = encoded_stimulus[:dummy_var_idxs[0]]
                        target_words = encoded_stimulus[dummy_var_idxs[0]+1:dummy_var_idxs[1]]
                        following_words = encoded_stimulus[dummy_var_idxs[1]+1:]   
                        
                    stimulus_cleaned = stimulus.replace("*","")
                    
                    output = openai.Completion.create(
                            engine = model_name,
                            prompt = stimulus_cleaned,
                            max_tokens = 0,
                            temperature = 0,
                            top_p = 1,
                            n = 1,
                            stream = False,
                            logprobs = 1,
                            stop = "\n",
                            echo = True
                            )
                    logprob = output.to_dict()['choices'][0].to_dict()['logprobs']
                    
                    surprisal_list = logprob["token_logprobs"][len(preceding_context):len(preceding_context)+len(target_words)]
                    
                    if surprisal_list[0]==None:
                        print("Problem with stimulus on line {0}: {1}\nCannot process the first token in a sentence/sequence.\n".format(str(j+1),stimulus_list[j]))
                    else:
                        sentence = tokenizer.decode(preceding_context+target_words)
                        target_string = "".join(logprob["tokens"][len(preceding_context):len(preceding_context)+len(target_words)])
                        surprisal = -np.log2(np.exp(np.sum(surprisal_list)))
                        num_tokens = len(target_words)
                        with open(filename,"a") as f:
                            f.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(
                            stimulus.replace("*",""),
                            sentence,
                            target_string,
                            surprisal,
                            num_tokens
                            ))
                except:
                    print("Problem with stimulus on line {0}: {1}\n".format(str(j+1),stimulus_list[j]))

def main():
    args = parse_args()


    try:
        output_directory,model_list,stimulus_file_list = process_args(args)
    except:
        "Error: Make sure you include arguments for the stimuli, output directory, GPT-3 models, and API key."
        return False


    try:
        run_models(output_directory,model_list,stimulus_file_list)
    except:
        print("Error: issue with stimuli, output directory, GPT-3 models chosen, or API key.")



if __name__ == "__main__":
    main()
