import json
import os
import torch
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
import argparse


def get_data_from_rebel_files(rebel_path, file_name):
    data = []
    with open(os.path.join(rebel_path, file_name), 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data



def main():
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebel_path", type=str, default="/home/finapolat/GenIE/data/rebel")
    parser.add_argument("--extraction_path", type=str, default="/home/finapolat/KGC-LLM/extractions")
    args = parser.parse_args()
    
    # template for an instrution with no example
    prompt = PromptTemplate(
        input_variables=["input_text"],
        template= """A triple has three components: subject, relations, object. Extract triples from the given text in the following format: ['subject', 'relation', 'object'] and put them in a list.
        Text to extract triples: {input_text} \n Extracted Triples: """)
    
    print("Loading model...")
    generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16,
                           trust_remote_code=True, device_map="auto", return_full_text=True)
    #generate_text = pipeline(model="google/flan-t5-large", 
       #                  torch_dtype=torch.bfloat16, trust_remote_code=True,  
        #                 device_map="auto")
                         
    hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
    llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
    print("Model loaded.")
    
    outfile = open(os.path.join(args.extraction_path, "zeroshot_extraction_dolly_on_REBEL.txt"), "w", encoding="utf-8")
    print("Loading data...")
    test_data = get_data_from_rebel_files(args.rebel_path, "en_test.jsonl")
    print("Data loaded.")
    print(f"Number of test instances: {len(test_data)}")
    
    print("Running zero-shot extraction...")
    #for d in test_data[:5]:
    for d in test_data:
        input_text = d["input"]
        extraction = llm_chain.run(input_text=input_text)
        extraction_dict = {"model_name": "google/flan-t5-large",
                           "dataset": "REBEL_en_test",
                           "input_text": input_text, 
                           "zero-shot extraction": extraction}
        #print(extraction_dict)
        outfile.write(json.dumps(extraction_dict) + "\n")
    outfile.close()
    print("Zero-shot extraction finished.")
    print(f"Extractions are written to the {args.extraction_path}.") 
    
if __name__ == "__main__":
    main()
    
    