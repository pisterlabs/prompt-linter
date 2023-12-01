from scipy.spatial.distance import cosine
import argparse
import json
import os
import openai
import pdb

def read_text(input_file):
    arr = open(input_file).read().split("\n")
    return arr[:-1]


class OpenAIModel:
    def __init__(self):
        self.debug = False
        self.model_name = None
        self.skip_key = True
        print("In OpenAI API constructor")


    def init_model(self,model_name = None):
        #print("OpenAI: Init model",model_name)
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if (openai.api_key == None):
            openai.api_key = ""
            print("API key not set")

        if (len(openai.api_key) == 0 and not self.skip_key):
                print("Open API key not set")
        
        if (model_name is None):
            self.model_name = "text-similarity-ada-001"
        else:
            self.model_name = model_name
        print("OpenAI: Init model complete",model_name)


    def compute_embeddings(self,input_file_name,input_data,is_file):
        if (len(openai.api_key) == 0 and not self.skip_key):
                print("Open API key not set")
                return [],[]
        #print("In compute embeddings after key check")
        in_file = self.model_name + '.'.join(input_file_name.split('.')[:-1]) + "_embed.json"
        cached = False
        try:
            fp = open(in_file)
            cached = True
            embeddings = json.load(fp)
            print("Using cached embeddings")
        except:
            pass
            
        texts = read_text(input_data) if is_file == True else input_data
        if (not cached):
            print(f"Computing embeddings for {input_file_name} and model {self.model_name}")
            response = openai.Embedding.create(
                input=texts,
                model=self.model_name
            )
            embeddings = []
            for i in range(len(response['data'])):
                embeddings.append(response['data'][i]['embedding'])
        if (not cached):
            with open(in_file,"w") as fp:
                json.dump(embeddings,fp)
        return texts,embeddings

    def output_results(self,output_file,texts,embeddings,main_index = 0):
        if (len(openai.api_key) == 0 and not self.skip_key):
                print("Open API key not set")
                return {}
        #print("In output results after key check")
        # Calculate cosine similarities
        # Cosine similarities are in [-1, 1]. Higher means more similar
        cosine_dict = {}
        #print("Total sentences",len(texts))
        for i in range(len(texts)):
                cosine_dict[texts[i]] = 1 - cosine(embeddings[main_index], embeddings[i])

        #print("Input sentence:",texts[main_index])
        sorted_dict = dict(sorted(cosine_dict.items(), key=lambda item: item[1],reverse = True))
        if (self.debug):
            for key in sorted_dict:
                print("Cosine similarity with  \"%s\" is: %.3f" % (key, sorted_dict[key]))
        if (output_file is not None):
            with open(output_file,"w") as fp:
                fp.write(json.dumps(sorted_dict,indent=0))
        return sorted_dict



if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='OpenAI model for sentence embeddings ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-input', action="store", dest="input",required=True,help="Input file with sentences")
        parser.add_argument('-output', action="store", dest="output",default="output.txt",help="Output file with results")
        parser.add_argument('-model', action="store", dest="model",default="text-similarity-ada-001",help="model name")

        results = parser.parse_args()
        obj = OpenAIModel()
        obj.init_model(results.model)
        texts, embeddings = obj.compute_embeddings(results.input,is_file = True)
        results = obj.output_results(results.output,texts,embeddings)
